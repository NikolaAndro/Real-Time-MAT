from MAT_models.ResNet_MAT import ResnetGenerator, set_init_counter, set_mixed_init, get_init_methods, reset_init_methods, weights_init
import torch
import torch.optim as optim
from learning.MAT_learning.TauGenerator import TauGenerator
def load_generators(crossover_versions,crossover_dimensions, config, args, gpulist, experiment):
    initial_tau_models = []
    weights_init_mix = False

    if args.arch_tau == 'resnet':
        # create N models with random weights and N optimizers
        for i in range(args.num_init_models):
            model_info = {}
            
            tau = TauGenerator(args, gpulist)
            tau.load_weights(i, args)
            tau.model = torch.nn.DataParallel(tau.model)
            tau.write_model_characteristics(args,crossover_versions,crossover_dimensions)
            tau.model.train(True)
            tau.model.module.cpu()
            tau.set_optimizer(args)
            

            # model is automatically on CUDA
            model = ResnetGenerator(3, 3, args.ngf, norm_type=args.norm_type, act_type=args.activ_func, gpu_ids=gpulist, n_blocks=6, use_dropout=args.dropout)    

            # In case of pretrained model
            if args.pretrained_generator_model_path:
                if args.debug_mode: print("\n\nLoading a pretrained GENERATOR model...\nPretrained model path: ",args.pretrained_generator_model_path)
                model.load_state_dict(torch.load(args.pretrained_generator_model_path)) 
            else:
                if i % 2 == 1:
                    weights_init_mix = True
                else:
                    weights_init_mix = False

                if weights_init_mix == True:
                    set_mixed_init(True)
                    model.apply(weights_init)
                else:
                    set_mixed_init(False)
                    init_method_id = 2 # for xavier_normal_
                    set_init_counter(init_method_id)
                    model.apply(weights_init)
            
            model_info['model_characteristics'] = {'model':'resnet18'}
            model_info['model_characteristics']['num_reproduction_models'] = args.num_reproduction_models
            model_info['model_characteristics']['crossover_versions'] = crossover_versions
            model_info['model_characteristics']['crossover_dimensions'] = crossover_dimensions
            model_info['model_characteristics']['greedy_approach'] = args.greedy_approach
            model_info['model_characteristics']['optimizer'] = args.optimizer
            model_info['model_characteristics']['lr'] = args.lr
            model_info['model_characteristics']['initialization_method(s)'] = get_init_methods()
            reset_init_methods()            

            model = torch.nn.DataParallel(model)
            model.train(True)

            # Create a deep copy of the model and then remove the original model so we don't have multiple objects referencing the same model.
            model_info['model'] = model#copy.deepcopy(model)
            model_info['model'].module.cpu()

            if model_info['model_characteristics']['optimizer'] == 'Adam':
                model_info['optimizer'] = optim.Adam(model_info['model'].parameters(), lr=args.lr, betas=(args.beta1, 0.999))
            elif model_info['model_characteristics']['optimizer'] == 'SGD':
                model_info['optimizer'] = optim.SGD(model_info['model'].parameters(), lr=args.lr,  momentum=0.9 , nesterov=True, dampening=0)
            elif model_info['model_characteristics']['optimizer'] == 'RMSprop':
                model_info['optimizer'] = optim.RMSprop(model_info['model'].parameters(), lr=args.lr, alpha=0.11, eps=1e-08, weight_decay=0,\
                     momentum=0, centered=False, foreach=None, maximize=False, differentiable=False)
            elif model_info['model_characteristics']['optimizer'] == 'Adagrad':
                model_info['optimizer'] = optim.Adagrad(model_info['model'].parameters(), lr=args.lr, lr_decay=0, weight_decay=0,\
                     initial_accumulator_value=0, eps=1e-10, foreach=None, maximize=False)
            elif model_info['model_characteristics']['optimizer'] == 'ASGD':
                model_info['optimizer'] = optim.ASGD(model_info['model'].parameters(), lr=args.lr, lambd=0.0001, alpha=0.75, t0=1000000.0,\
                     weight_decay=0, foreach=None, maximize=False)

            # This doesn’t increase the amount of GPU memory available for PyTorch. 
            # However, it may help reduce fragmentation of GPU memory in certain cases. 
            torch.cuda.empty_cache()

            initial_tau_models.append(tau)
            
            # reoprt the model information to comet
            if args.comet:
                for key, val in  tau.model_characteristics.items():
                    experiment.log_parameter("model_" + str(i) + "_" + str(key), val)
    else:
        raise("Currently, 'resnet' is the only option available.")

    return initial_tau_models

def load_mtask_model(args):
    if args.arch == 'resnet-18':
        from mtask_models.taskonomy_models import resnet18_taskonomy
        mtask_model = resnet18_taskonomy(pretrained=False, tasks=args.task_set)

    elif args.arch == 'resnet-50':
        from mtask_models.taskonomy_models import resnet50_taskonomy
        mtask_model = resnet50_taskonomy(pretrained=False, tasks=args.task_set)

    elif args.arch == 'wide_resnet-50':
        from mtask_models.taskonomy_models import wide_resnet50_2
        mtask_model = wide_resnet50_2(pretrained=False, tasks=args.task_set)

    elif args.arch == 'wide_resnet-101':
        from mtask_models.taskonomy_models import wide_resnet101_2
        mtask_model = wide_resnet101_2(pretrained=False, tasks=args.task_set)

    elif args.arch == 'resnet-152':
        from mtask_models.taskonomy_models import resnet152_taskonomy
        mtask_model = resnet152_taskonomy(pretrained=False, tasks=args.task_set)

    elif args.arch == 'xception':
        from mtask_models.xception_taskonomy_small import xception_taskonomy_small
        mtask_model = xception_taskonomy_small(pretrained=False, tasks=args.task_set)

    elif args.arch == 'xception-full':
        from mtask_models.xception_taskonomy_new import xception_taskonomy_new
        mtask_model = xception_taskonomy_new(pretrained=False, tasks=args.task_set)

    elif 'drn' in args.arch:
        # CHANGE HERE FOR CITYSCAPE
        from mtask_models.DRNSegDepth import DRNSegDepth
        mtask_model = DRNSegDepth(args.arch,
                            classes=19,
                            pretrained_model=None,
                            pretrained=False,
                            tasks=args.task_set)
    
    # In case of pretrained model
    if args.pretrained_mtask_model_path:
        m = torch.load(args.pretrained_mtask_model_path)
        mtask_model.load_state_dict(m['state_dict']) 
 
    mtask_model = torch.nn.DataParallel(mtask_model)
    mtask_model.train()

    return mtask_model

