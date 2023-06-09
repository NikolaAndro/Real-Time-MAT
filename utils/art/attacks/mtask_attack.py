import io
from torch.autograd import Variable
from torch import nn
import torch
from learning.utils_learn import forward_transform, back_transform, fast_hist, clamp_tensor, per_class_iu
import numpy as np
from utils.vulnerability import compute_vulnerability, get_second_order_grad

batch_log = 10

class AverageTimeSeries(object):
    """Computes and stores the average and current value of a time serie"""

    def __init__(self):
            self.reset()

    def reset(self, axis=0):
            self.keys = None
            self.axis = axis
            self.series = []

    def update(self, val):
        if self.keys is None:
                keys = list(val[0].keys())
                self.keys = [a[:a.rfind("_")] for a in keys]

        self.series.append([np.fromiter(v.values(), dtype=float) for v in val])

    @property
    def count(self):
            return len(self.series)

    @property
    def avg(self):
        try:
            return np.array(self.series).mean(axis=self.axis)
        except Exception as e:
            print(e)
            print(self.series)

    def mean(self, axis):
            return np.array(self.series).mean(axis=axis)



def error(a,b):
    return torch.abs(a - b).sum().data/a.shape[0]

def mse(a,b):
    return ((a - b)*(a - b)).sum().data/a.shape[0]

def get_torch_std(info):
    std_array = np.asarray(info["std"])
    tensor_std = torch.from_numpy(std_array)
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()
    return tensor_std

"""
def PGD_attack_mtask_L2(x, y, mask, net, criterion, task_name, epsilon, steps, dataset, step_size):
    net.eval()

    # tensor_std = get_torch_std(info)
    if epsilon == 0:
        return Variable(x, requires_grad=False)

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    rescale_term = 2./255
    epsilon = epsilon * rescale_term
    step_size = step_size * rescale_term #TODO: may need this if results not good

    x_adv = x.clone()
    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        x = x.cuda()
        x_adv = x_adv.cuda()
        for keys, m in mask.items():
            mask[keys] = m.cuda()
        for keys, tar in y.items():
            y[keys] = tar.cuda()

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        h_adv = net(x_adv)

        grad_total_loss = None
        for each in task_name:
            if grad_total_loss is None:
                grad_total_loss = criterion[each](h_adv[each], y[each], mask[each])
            else:
                grad_total_loss = grad_total_loss + criterion[each]

        net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        grad_total_loss.backward()

        grad = x_adv.grad

        # grad_normalized = grad / np.linalg.norm(grad)
        # print('epsilon', epsilon)
        x_adv = x_adv + grad * epsilon
        x_delta = x_adv - x
        x_delta_normalized = x_delta / torch.norm(x_delta, 2)

        x_adv = x + x_delta_normalized * epsilon

        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        #TODO: volatile option for backward, check later

    return x_adv
"""

def PGD_attack_mtask(x, y, mask, net, criterion, task_name, epsilon, steps, step_size, args, comet=None,
                     batch_index = None,strategy=None,using_noise=True,norm = 'Linf' ):
    net.eval()

    if batch_index % batch_log ==0:
        print("attacking batch {batch_index} on {task_name} with strategy {strategy}".format(batch_index=batch_index, task_name=task_name, strategy=strategy))
    # tensor_std = get_torch_std(info)
    if epsilon == 0:
        return Variable(x, requires_grad=False)

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    rescale_term = 2./255
    epsilon = epsilon * rescale_term
    step_size = step_size * rescale_term

    # print('epsilon', epsilon, epsilon / rescale_term)
    GRID_WEIGTHS = [10**i for i in range(-9,9)]
    weights = {k:[] for k in task_name}

    search_iters = np.power(len(GRID_WEIGTHS), len(task_name)) if strategy == "GRID_SEARCH" else 1
    best_advs = {"score": -np.inf, "adv": None, "metrics":[]}
    strtg = strategy.split("_")

    best_advs["metric"] = task_name[int(strtg[-1])] if len(strtg)>2 else task_name[0]

    for j in range(search_iters):
        x_adv = x.clone()

        pert_upper = x_adv + epsilon
        pert_lower = x_adv - epsilon

        upper_bound = torch.ones_like(x_adv)
        lower_bound = -torch.ones_like(x_adv)

        upper_bound = torch.min(upper_bound, pert_upper)
        lower_bound = torch.max(lower_bound, pert_lower)

        ones_x = torch.ones_like(x).float()
        if GPU_flag:

            x_adv = x_adv.cuda()
            upper_bound = upper_bound.cuda()
            lower_bound = lower_bound.cuda()
            for keys, m in mask.items():
                mask[keys] = m.cuda()
            for keys, tar in y.items():
                y[keys] = tar.cuda()


        if using_noise:
            noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
            if GPU_flag:
                noise = noise.cuda()
            x_adv = x_adv + noise
            x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

        x_adv = Variable(x_adv, requires_grad=True)
        base_loss = {}
        init_loss = {}
        metrics = []

        if args.store_examples > 0 and batch_index % args.store_examples == 0 and comet is not None:
            buffer_adv = io.BytesIO()
            torch.save(x.detach(), buffer_adv)
            comet.log_asset_data(buffer_adv.getvalue(), step=0, epoch=batch_index,
                                 name="clean_batch_{}".format(batch_index))

        for i in range(steps):
            h_adv = net(x_adv)

            grad_total_loss = None
            task_metrics = {}
            nb_tasks = len(task_name)
            for k, each in enumerate(task_name):

                task_metrics['Loss {}_{}'.format(each,batch_index)] = criterion[each](h_adv[each], y[each], mask[each]).cpu().detach().item()

                #comet.log_metric('Loss {}'.format(each), criterion[each](h_adv[each], y[each], mask[each]).item(),
                                 #step=i,epoch=batch_index )

                if each =="segmentsemantic":
                    class_prediction = torch.argmax(h_adv['segmentsemantic'], dim=1)
                    class_prediction = class_prediction.cpu().data.numpy() if torch.cuda.is_available() else class_prediction.data.numpy()
                    target_seg = y['segmentsemantic'].cpu().data.numpy() if torch.cuda.is_available() else y[
                        'segmentsemantic'].data.numpy()
                    hist = fast_hist(class_prediction.flatten(), target_seg.flatten(), args.classes)
                    adv_ious = per_class_iu(hist) * 100
                    adv_mIoU = round(np.nanmean(adv_ious), 2)
                    task_metrics['mIoU Segment_{}'.format(batch_index)]=adv_mIoU
                    #comet.log_metric('mIoU Segment', adv_mIoU,step=i, epoch=batch_index)
                else:
                    adv_mse = mse(h_adv[each], y[each])
                    task_metrics['MSE {}_{}'.format(each,batch_index)] = adv_mse.cpu().detach().item()
                    #comet.log_metric('MSE {}'.format(each), adv_mse, step=i, epoch=batch_index)

                if strategy=="GRID_SEARCH":
                    w = GRID_WEIGTHS[j%np.power(len(task_name),k+1)//k]
                    weight = w* criterion[each](h_adv[each], y[each], mask[each]).cpu().detach().item()

                    task_metrics['w_Loss {}_{}'.format(each,batch_index)] = weight
                    if grad_total_loss is None:
                        grad_total_loss = w* criterion[each](h_adv[each], y[each], mask[each])
                    else:
                        grad_total_loss = grad_total_loss + w* criterion[each](h_adv[each], y[each], mask[each])

                elif strategy=="RELATIVE_WEIGHT" or strategy=="CYCLIC_STEP" or strategy=="CYCLIC_BATCH":
                    if base_loss.get(each, None) is None:

                        base_loss[each] = criterion[each](h_adv[each], y[each], mask[each]).cpu().detach().item()
                        task_metrics['w_Loss {}_{}'.format(each, batch_index)] = base_loss[each]


                    if (strategy=="CYCLIC_STEP" and i % nb_tasks != k) or (strategy=="CYCLIC_BATCH" and batch_index % nb_tasks != k):
                        print("skipping as strategy {} step {} batch {} task {}".format(strategy,i,batch_index, k))

                    elif strategy=="CYCLIC_STEP" or strategy=="CYCLIC_BATCH" or strategy=="RELATIVE_WEIGHT":

                        if init_loss.get(each, None) is None:
                            init_loss[each] = True

                            if grad_total_loss is None:
                                grad_total_loss = criterion[each](h_adv[each], y[each], mask[each])
                            else:
                                grad_total_loss = grad_total_loss + criterion[each](h_adv[each], y[each], mask[each])

                        else:
                            relative_loss = criterion[each](h_adv[each], y[each], mask[each]).cpu().detach().item() / \
                                            base_loss[each]
                            task_metrics['w_Loss {}_{}'.format(each, batch_index)] = relative_loss

                            if grad_total_loss is None:
                                grad_total_loss = criterion[each](h_adv[each], y[each], mask[each]) / base_loss[each]
                            else:
                                grad_total_loss = grad_total_loss + criterion[each](h_adv[each], y[each], mask[each]) / \
                                                  base_loss[each]


                else:
                    relative_loss = criterion[each](h_adv[each], y[each], mask[each]).cpu().detach().item()
                    task_metrics['w_Loss {}_{}'.format(each, batch_index)] = relative_loss

                    if grad_total_loss is None:
                        grad_total_loss = criterion[each](h_adv[each], y[each], mask[each])
                    else:
                        grad_total_loss = grad_total_loss + criterion[each](h_adv[each], y[each], mask[each])

            metrics.append(task_metrics)

            net.zero_grad()

            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)

            grad_total_loss.backward()

            if norm == 'Linf':
                x_adv.grad.sign_()
                x_adv = x_adv + step_size * x_adv.grad
                x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
            else:
                # grad_normalized = grad / np.linalg.norm(grad)
                # print('epsilon', epsilon)
                x_adv = x_adv + x_adv.grad * epsilon
                x_delta = x_adv - x.cuda()
                x_delta_normalized = x_delta / torch.norm(x_delta, 2)

                x_adv = x.cuda() + x_delta_normalized * epsilon

            if args.metrics.find("hessian") > -1:
                hessian = get_second_order_grad(x_adv,net)
                for (k,v) in hessian.items():
                    task_metrics['hessian {}_{}'.format(k,batch_index)] = v

            if args.store_examples > 0 and batch_index%args.store_examples==0 and comet is not None:
                #buffer_clean = io.BytesIO()
                buffer_adv = io.BytesIO()
                #torch.save(x.detach(), buffer_clean)
                torch.save(x_adv.detach(), buffer_adv)
                comet.log_asset_data(buffer_adv.getvalue(),step=i, epoch=batch_index, name="adv_batch_{}".format(batch_index))

            if args.metrics.find("vuln") >-1:
                vuln = compute_vulnerability(x,x_adv,net, criterion, y , mask) #batch_index
                for (k,v) in vuln.items():
                    task_metrics['{}_{}'.format(k,batch_index)] = v


            x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
            #TODO: volatile option for backward, check later

        if strategy[:11]!="GRID_SEARCH":
            if comet is not None:
                for im, m in enumerate(metrics):
                    comet.log_metrics(m, step=im)

            return x_adv, metrics

        h_adv = net(x_adv)
        adv_this_mse = dict([(each, mse(h_adv[each].float(), y[each])) for each in task_name])
        if adv_this_mse[best_advs["metric"]] > best_advs["score"]:
            best_advs["score"] = adv_this_mse[best_advs["metric"]]
            best_advs["adv"] = x_adv
            best_advs["metrics"] =  metrics
            print("new best score {} at {}".format(best_advs["score"], j))

    if comet is not None:
        for im, m in enumerate(best_advs["metrics"]):
            comet.log_metrics(m, step=im)

    return best_advs["adv"], best_advs["metrics"]


def PGD_attack_mtask_city(x, y, mask, net, criterion, task_name, epsilon, steps, dataset, step_size, info, args, using_noise=True):
    # print('crop ', torch.max(x), torch.min(x))
    # print('size', x.size())

    # std_array = np.asarray(info["std"])
    # tensor_std = torch.from_numpy(std_array)
    # tensor_std = tensor_std.unsqueeze(0)
    # tensor_std = tensor_std.unsqueeze(2)
    # tensor_std = tensor_std.unsqueeze(2).float()
    tensor_std = get_torch_std(info)

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    x_adv = x.clone()

    epsilon = epsilon / 255.
    step_size = step_size / 255.

    pert_epsilon = torch.ones_like(x_adv) * epsilon / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon


    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = 0

        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()

        for keys, m in mask.items():
            mask[keys] = m.cuda()
        for keys, tar in y.items():
            y[keys] = tar.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    if using_noise:
        noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
        if GPU_flag:
            noise = noise.cuda()
        noise = noise / tensor_std
        x_adv = x_adv + noise
        x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        h_adv = net(x_adv)
        grad_total_loss = None
        for each in task_name:
            if grad_total_loss is None:
                grad_total_loss = criterion[each](h_adv[each], y[each], mask[each])
            else:
                grad_total_loss = grad_total_loss + criterion[each](h_adv[each], y[each], mask[each])

        # # elif dataset == 'ade20k':
        # #     h_adv = net(x_adv,segSize = (256,256))
        #
        # # total_loss = 0
        # # for keys, loss_func in criterion:
        # #     if keys in task_name:
        # #         loss = loss_func(h_adv[keys], y[keys])
        # #         total_loss += loss
        #
        # first_loss = None
        # loss_dict = {}
        # for c_name, criterion_fun in criterion.items():
        #     if first_loss is None:
        #         first_loss = c_name
        #         # print('l output target', output)
        #         # print('ratget', target)
        #         loss_dict[c_name] = criterion_fun(h_adv, y)
        #         # print('caname', c_name, loss_dict[c_name])
        #     else:
        #         loss_dict[c_name] = criterion_fun(h_adv[c_name], y[c_name])
        #
        # grad_total_loss = None
        # for each in args.test_task_set:
        #     if grad_total_loss is None:
        #         grad_total_loss = loss_dict[each]
        #     else:
        #         grad_total_loss = grad_total_loss + loss_dict[each]



        # cost = Loss(h_adv[0], y) #TODO: works, but is this the correct place to convert to long??
        #print(str(i) + ': ' + str(cost.data))
        net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        grad_total_loss.backward()

        x_adv.grad.sign_()
        x_adv = x_adv + step_size_tensor * x_adv.grad
        #print(x_adv.data[:,4,4])
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        # x_adv = torch.where(x_adv > upper_bound, upper_bound, x_adv)
        # x_adv = torch.where(x_adv < lower_bound, lower_bound, x_adv)
        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        #TODO: volatile option for backward, check later

    # sample =x_adv.data
    # im_rgb = back_transform(sample, info)[0]
    # im_rgb = np.moveaxis(im_rgb.cpu().numpy().squeeze(), 0, 2)
    #
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(im_rgb)
    # plt.show()

    return x_adv

# def PGD_drnseg_masked_attack_city(image_var,label,mask,attack_mask,model,criteria,tasks,
#                                                              args.epsilon,args.steps,args.dataset,
#                                                              args.step_size,info,args,using_noise=True):
def PGD_drnseg_masked_attack_city(x, y, attack_mask, net, criterion, epsilon, steps, dataset, step_size, info, args, using_noise=True):
    tensor_std = get_torch_std(info)

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag = True

    x_adv = x.clone()

    epsilon = epsilon / 255.
    step_size = step_size / 255.

    ones_like_x_adv = torch.ones_like(x_adv)

    if GPU_flag:
        ones_like_x_adv = ones_like_x_adv.cuda()
        tensor_std = tensor_std.cuda()

    pert_epsilon = ones_like_x_adv * epsilon / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon

    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    # TODO: print and check the bound

    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        # Loss = 0

        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()

        y = y.cuda()

        # for keys, m in mask.items():
        #     mask[keys] = m.cuda()
        # for keys, tar in y.items():
        #     y[keys] = tar.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        #if i==10:
        #    print('PGD_drnseg_masked_attack_city attack step', i)
        h_adv = net(x_adv)  # dict{rep:float32,segmentasemantic:float32, depth_zbuffer:float32, reconstruct:float32}
        grad_total_loss = None
        # print("Task names ", task_name)
        # for each in task_name:
            # print("IN ",each)
            # if grad_total_loss is None:
                # print(each,y.keys(),h_adv[1])
                # print(h_adv)
        ignore_value = 255
                # print(mask[each].type(), attack_mask.type())
        attack_mask = attack_mask.long()
        # mask_each = mask[each]  # segmentsemantic is long and others are float.
        # mask_total = mask_each * attack_mask  # attack_mask is float, mask_total is float.
        # mask_total = mask_total.long()
                # print(each, (y[each] * mask_total).type())
                # print((ignore_value * (1-mask_total)).type()) # types(str, )
        y = y * attack_mask + ignore_value * (1 - attack_mask)  # y is {auto:float,segsem:int64,deoth:float}
        grad_total_loss = criterion(h_adv[0], y)

        net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        grad_total_loss.backward()

        x_adv.grad.sign_()
        x_adv = x_adv + step_size_tensor * x_adv.grad
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        x_adv = Variable(x_adv.data, requires_grad=True)  # TODO: optimize, remove this variable init each

    return x_adv


def PGD_masked_attack_mtask_city(x, y, mask, attack_mask, net, criterion, task_name, epsilon, steps, dataset, step_size, info, args, using_noise=True):
    # print('crop ', torch.max(x), torch.min(x))
    # print('size', x.size())

    # std_array = np.asarray(info["std"])
    # tensor_std = torch.from_numpy(std_array)
    # tensor_std = tensor_std.unsqueeze(0)
    # tensor_std = tensor_std.unsqueeze(2)
    # tensor_std = tensor_std.unsqueeze(2).float()
    tensor_std = get_torch_std(info)

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    x_adv = x.clone()

    epsilon = epsilon / 255.
    step_size = step_size / 255.
    
    ones_like_x_adv = torch.ones_like(x_adv) 
    
    if GPU_flag:
        ones_like_x_adv = ones_like_x_adv.cuda()
        tensor_std = tensor_std.cuda()

    ones_like_x_adv = torch.ones_like(x_adv)

    if GPU_flag:
        ones_like_x_adv = ones_like_x_adv.cuda()

    pert_epsilon = ones_like_x_adv * epsilon / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon

    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = 0

        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()

        for keys, m in mask.items():
            mask[keys] = m.cuda()
        for keys, tar in y.items():
            y[keys] = tar.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        h_adv = net(x_adv) # dict{rep:float32,segmentasemantic:float32, depth_zbuffer:float32, reconstruct:float32}
        grad_total_loss = None
        # print("Task names ", task_name)
        for each in task_name:
            # print("IN ",each)
            if grad_total_loss is None:
                # print(each,y.keys(),h_adv[1])
                # print(h_adv)
                ignore_value = 255
                # print(mask[each].type(), attack_mask.type())
                attack_mask = attack_mask.long()
                mask_each =  mask[each] #segmentsemantic is long and others are float.
                mask_total = mask_each * attack_mask # attack_mask is float, mask_total is float.
                mask_total = mask_total.long()
                # print(each, (y[each] * mask_total).type())
                # print((ignore_value * (1-mask_total)).type()) # types(str, )
                y[each] = y[each] * mask_total + ignore_value * (1-mask_total) # y is {auto:float,segsem:int64,deoth:float}
                grad_total_loss = criterion[each](h_adv[each], y[each], mask[each]*attack_mask)
            else:
                grad_total_loss = grad_total_loss + criterion[each](h_adv[each], y[each], mask[each]*attack_mask)

        net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        grad_total_loss.backward()

        x_adv.grad.sign_()
        x_adv = x_adv + step_size_tensor * x_adv.grad
        #print(x_adv.data[:,4,4])
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        # x_adv = torch.where(x_adv > upper_bound, upper_bound, x_adv)
        # x_adv = torch.where(x_adv < lower_bound, lower_bound, x_adv)
        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        #TODO: volatile option for backward, check later

    return x_adv


def PGD_attack(x, y, net, Loss, epsilon, steps, dataset, step_size, info, using_noise=True):
    # print('crop ', torch.max(x), torch.min(x))
    # print('size', x.size())

    std_array = np.asarray(info["std"])
    tensor_std = torch.from_numpy(std_array)
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()



    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    x_adv = x.clone()

    pert_epsilon = torch.ones_like(x_adv) * epsilon / tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon


    upper_bound = torch.ones_like(x_adv)
    lower_bound = torch.zeros_like(x_adv)

    upper_bound = forward_transform(upper_bound, info)
    lower_bound = forward_transform(lower_bound, info)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    #TODO: print and check the bound


    ones_x = torch.ones_like(x).float()
    if GPU_flag:
        Loss = Loss.cuda()
        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        tensor_std = tensor_std.cuda()
        ones_x = ones_x.cuda()
        y = y.cuda()

    step_size_tensor = ones_x * step_size / tensor_std

    if using_noise:
        noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
        if GPU_flag:
            noise = noise.cuda()
        noise = noise / tensor_std
        x_adv = x_adv + noise
        x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        h_adv = net(x_adv)
        # elif dataset == 'ade20k':
        #     h_adv = net(x_adv,segSize = (256,256))
        cost = Loss(h_adv[0], y) #TODO: works, but is this the correct place to convert to long??
        #print(str(i) + ': ' + str(cost.data))
        net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv + step_size_tensor * x_adv.grad
        #print(x_adv.data[:,4,4])
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)
        # x_adv = torch.where(x_adv > upper_bound, upper_bound, x_adv)
        # x_adv = torch.where(x_adv < lower_bound, lower_bound, x_adv)
        x_adv = Variable(x_adv.data, requires_grad=True)  #TODO: optimize, remove this variable init each
        #TODO: volatile option for backward, check later

    return x_adv



