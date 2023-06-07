from distutils.command.clean import clean
import numpy as np
import torch

from utils.art.attacks.mtask_attack import PGD_attack_mtask, PGD_attack_mtask_city, mse, error, AverageTimeSeries
from learning.utils_learn import accuracy, AverageMeter, fast_hist, back_transform, per_class_iu
from dataloaders.utils import decode_segmap

import logging

import time

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

batch_logging = 5

def mtask_forone_advacc(val_loader, model, criterion, task_names, args, info, epoch=0, writer=None,
                        comet=None, test_flag=False, test_vis=False, norm='Linf'):
    """
    NOTE: test_flag is for the case when we are testing for multiple models, need to return something to be able to plot and analyse
    """
    assert len(task_names) > 0
    avg_losses = {}
    adv_avg_losses = {}

    avg_errors = {}
    adv_avg_errors = {}

    avg_mse = {}
    adv_avg_mse = {}

    num_classes = args.classes
    hist = np.zeros((num_classes, num_classes))
    adv_hist = np.zeros((num_classes, num_classes))

    for c_name, criterion_fun in criterion.items():
        avg_losses[c_name] = AverageMeter()
        adv_avg_losses[c_name] = AverageMeter()

        avg_errors[c_name] = AverageMeter()
        adv_avg_errors[c_name] = AverageMeter()

        avg_mse[c_name] = AverageMeter()
        adv_avg_mse[c_name] = AverageMeter()

    seg_accuracy = AverageMeter()
    seg_clean_accuracy = AverageMeter()
    step_metrics = AverageTimeSeries()

    # need to set my own accuracy here

    model.eval() # this is super important for correct including the batchnorm

    print("using norm type", norm)

    for i, (input_batch, target_batch, mask_batch) in enumerate(val_loader):
        # print("\niteration in dataloader iterator\n")
        if args.debug:
            if i>0:
                break
        # print(len(input))
        # print(len(target))
        # print(len(mask))
        # exit()
        if 'segmentsemantic' in criterion.keys():
            # rep/trunk, task1, task2 ... task_numTasaks
            clean_output = model(torch.autograd.Variable(input_batch.cuda(), requires_grad=False))
            # print(len(clean_output))
            # print(type(clean_output))
            # for i in clean_output:
            #     print(i)

            # print("****** results ******")
            # print(len(clean_output['segmentsemantic']))
            # print(len(clean_output['depth_euclidean']))

            # print("****** target ******")
            # print(len(target['segmentsemantic'].long()))
            # print(len(target['depth_euclidean'].long()))

            # print("****** accuracy ******")

            # print(accuracy(clean_output['segmentsemantic'], target['segmentsemantic'].long().cuda()))
            # print(accuracy(clean_output['depth_euclidean'], target['depth_euclidean'].long().cuda()))

            # print("***** mask *****")
            # print(len(mask['depth_euclidean']))
            # print(len(mask['segmentsemantic']))
            # exit()
            seg_clean_accuracy.update(accuracy(clean_output['segmentsemantic'], target_batch['segmentsemantic'].long().cuda()),
                                      input_batch.size(0))
            
        if args.steps == 0 or args.step_size == 0:
            args.epsilon = 0
        
        if args.dataset == 'taskonomy':
            pre_napada = time.time()
            adv_img, batch_metrics = PGD_attack_mtask(input_batch, target_batch, mask_batch, model, criterion, task_names, args.epsilon,
                                      args.steps,args.step_size, args,batch_index=i, using_noise=True, norm=norm,
                                       comet=comet if i % batch_logging == 0 else None, strategy=args.strategy)
            ukupno_vreme = time.time() - pre_napada
            print("Ukupno vreme napada na jednu fotografiju je: ",ukupno_vreme)

            print(len(adv_img))
            # print(batch_metrics)
            step_metrics.update(batch_metrics)
            
            #NIKOLA : This is where I need to call MAT ATTACK
        
        elif args.dataset == 'cityscape':
            adv_img = PGD_attack_mtask_city(input_batch, target_batch, mask_batch, model, criterion, task_names, args.epsilon, args.steps,
                                       args.dataset,
                                       args.step_size, info, args, using_noise=True)
        #elif norm == 'l2':
        #    adv_img = PGD_attack_mtask_L2(input, target, mask, model, criterion, task_name, args.epsilon, args.steps,
        #                               args.dataset,
        #                               args.step_size)
        # image_var = torch.autograd.Variable(adv_img.data, requires_grad=False)
        image_var = input_batch.data
        adv_image_var = adv_img.data
        # image_var = input
        if torch.cuda.is_available():
            image_var = image_var.cuda()
            adv_image_var = adv_image_var.cuda()
            for keys, m in mask_batch.items():
                mask_batch[keys] = m.cuda()
            for keys, tar in target_batch.items():
                target_batch[keys] = tar.cuda()

        # print("diff", torch.sum(torch.abs(raw_input-image_var)))
        #NIKOLA: forward pass / evaluate
        with torch.no_grad():
            output = model(image_var)
            adv_output = model(adv_image_var)


        for key,val in adv_output.items():
            print(key, end=" ")
            # if key == "depth_euclidean":
            #     print(val)
            print(val.shape)
       
        sum_loss = None
        loss_dict = {}

        adv_sum_loss = None
        adv_loss_dict = {}

        for c_name, criterion_fun in criterion.items():
            this_loss = criterion_fun(output[c_name].float(), target_batch[c_name],
                                      mask_batch[c_name])

            adv_this_loss = criterion_fun(adv_output[c_name].float(), target_batch[c_name],
                                      mask_batch[c_name])



            this_error = error(output[c_name].float(), target_batch[c_name])

            adv_this_error = error(adv_output[c_name].float(), target_batch[c_name])

            this_mse = mse(output[c_name].float(), target_batch[c_name])

            adv_this_mse = mse(adv_output[c_name].float(), target_batch[c_name])

            if sum_loss is None:
                sum_loss = this_loss
                adv_sum_loss = adv_this_loss

            else:
                sum_loss = sum_loss + this_loss
                adv_sum_loss = adv_sum_loss + adv_this_loss

            loss_dict[c_name] = this_loss
            adv_loss_dict[c_name] = adv_this_loss

            avg_losses[c_name].update(loss_dict[c_name].data.item(), input_batch.size(0))
            adv_avg_losses[c_name].update(adv_loss_dict[c_name].data.item(), input_batch.size(0))

            avg_errors[c_name].update(this_error, input_batch.size(0))
            adv_avg_errors[c_name].update(adv_this_error, input_batch.size(0))

            avg_mse[c_name].update(this_mse, input_batch.size(0))
            adv_avg_mse[c_name].update(adv_this_mse, input_batch.size(0))

        if test_vis and i % batch_logging == 0:

            if writer is not None:
                writer.add_image('Val/image clean ', back_transform(input_batch, info)[0])
                writer.add_image('Val/image adv ', back_transform(adv_img, info)[0])
            if comet is not None:
                comet.log_image(back_transform(input_batch, info)[0].cpu(), name='Val/image clean ', image_channels='first', step=i)
                comet.log_image(back_transform(adv_img, info)[0].cpu(), name='Val/image adv ', image_channels='first', step=i)


        if 'segmentsemantic' in criterion.keys():
            # this is accuracy for segmentation
            seg_accuracy.update(accuracy(adv_output['segmentsemantic'], target_batch['segmentsemantic'].long()), input_batch.size(0))
            target_seg = target_batch['segmentsemantic'].cpu().data.numpy() if torch.cuda.is_available() else target_batch[
                'segmentsemantic'].data.numpy()

            class_prediction = torch.argmax(output['segmentsemantic'], dim=1)
            class_prediction = class_prediction.cpu().data.numpy() if torch.cuda.is_available() else class_prediction.data.numpy()
            hist += fast_hist(class_prediction.flatten(), target_seg.flatten(), num_classes)

            adv_prediction = torch.argmax(adv_output['segmentsemantic'], dim=1)
            adv_prediction = adv_prediction.cpu().data.numpy() if torch.cuda.is_available() else adv_prediction.data.numpy()
            hist += fast_hist(class_prediction.flatten(), target_seg.flatten(), num_classes)
            adv_hist += fast_hist(adv_prediction.flatten(), target_seg.flatten(), num_classes)

            if i % batch_logging == 0:
                class_prediction = torch.argmax(output['segmentsemantic'], dim=1)
                adv_class_prediction = torch.argmax(adv_output['segmentsemantic'], dim=1)
                # print(target['segmentsemantic'].shape)
                decoded_target = decode_segmap(
                    target_batch['segmentsemantic'][0][0].cpu().data.numpy() if torch.cuda.is_available() else
                    target_batch['segmentsemantic'][0][0].data.numpy(),
                    args.dataset)
                decoded_target = np.moveaxis(decoded_target, 2, 0)

                decoded_class_prediction = decode_segmap(
                    class_prediction[0].cpu().data.numpy() if torch.cuda.is_available() else class_prediction[
                        0].data.numpy(), args.dataset)
                decoded_class_prediction = np.moveaxis(decoded_class_prediction, 2, 0)

                adv_decoded_class_prediction = decode_segmap(
                    adv_class_prediction[0].cpu().data.numpy() if torch.cuda.is_available() else adv_class_prediction[
                        0].data.numpy(), args.dataset)
                adv_decoded_class_prediction = np.moveaxis(adv_decoded_class_prediction, 2, 0)

                if test_vis:

                    if writer is not None:
                        writer.add_image('Val/image gt for adv ', decoded_target)
                        writer.add_image('Val/image clean prediction ', decoded_class_prediction)
                        writer.add_image('Val/image adv prediction ', adv_decoded_class_prediction)
                    if comet is not None:

                        comet.log_image(decoded_target,                           name='Val/image gt for adv ',
                                        image_channels='first', step=i)
                        comet.log_image(decoded_class_prediction, name='Val/image clean prediction ',
                                        image_channels='first', step=i)
                        comet.log_image(adv_decoded_class_prediction,                 name='Val/image adv prediction ',
                                        image_channels='first', step=i)

        if 'segmentsemantic' in criterion.keys():
            print("clean seg accuracy: {}".format(seg_clean_accuracy.avg))
            if comet is not None:
                comet.log_metric('segmentsemantic Clean Score', seg_clean_accuracy.avg, step=i)

        break

    str_attack_result = ''
    str_not_attacked_task_result = ''
    for keys, loss_term in criterion.items():
        if keys in task_names:
            str_attack_result += 'Attacked Loss: {} {loss.val:.4f} ({loss.avg:.4f})\t'.format(keys, loss=avg_losses[keys])
        else:
            str_not_attacked_task_result += 'Not att Task Loss: {} {loss.val:.4f} ({loss.avg:.4f})\t'.format(keys, loss=avg_losses[keys])


    print('clean task')
    print(str_not_attacked_task_result)
    # Tensorboard logger
    if not test_flag:

        if comet is not None:
            print(np.array(step_metrics.series).shape)
            avg = step_metrics.avg
            for step in range(avg.shape[0]):
                metrics = dict(zip(step_metrics.keys, avg[step]))
                comet.log_metrics(metrics, step=step)

        for keys, _ in criterion.items():
            if keys in task_names:
                    if writer is not None:
                        writer.add_scalar('Val Adv Attacked Task/ Avg clean Loss {}'.format(keys), avg_losses[keys].avg,
                                          epoch)
                        writer.add_scalar('Val Adv Attacked Task/ Avg clean MAE {}'.format(keys), avg_errors[keys].avg,
                                          epoch)
                        writer.add_scalar('Val Adv Attacked Task/ Avg clean MSE {}'.format(keys), avg_mse[keys].avg, epoch)

                        writer.add_scalar('Val Adv Attacked Task/ Avg adv Loss {}'.format(keys), avg_losses[keys].avg,
                                          epoch)
                        writer.add_scalar('Val Adv Attacked Task/ Avg adv MAE {}'.format(keys), avg_errors[keys].avg,
                                          epoch)
                        writer.add_scalar('Val Adv Attacked Task/ Avg adv MSE {}'.format(keys), avg_mse[keys].avg, epoch)

                    if comet is not None:

                        comet.log_metric('Val Adv Attacked Task/ Avg clean MAE {}'.format(keys), avg_errors[keys].avg)
                        comet.log_metric('Val Adv Attacked Task/ Avg clean Loss {}'.format(keys), avg_losses[keys].avg)
                        comet.log_metric('Val Adv Attacked Task/ Avg clean MSE {}'.format(keys), avg_mse[keys].avg)
                        comet.log_metric('Val Adv Attacked Task/ Avg adv Loss {}'.format(keys), adv_avg_losses[keys].avg)
                        comet.log_metric('Val Adv Attacked Task/ Avg adv MAE {}'.format(keys), adv_avg_errors[keys].avg)
                        comet.log_metric('Val Adv Attacked Task/ Avg adv MSE {}'.format(keys), adv_avg_mse[keys].avg)
            else:
                    if writer is not None:
                        writer.add_scalar('Val Adv  not attacked Task/ Avg Loss {}'.format(keys), avg_losses[keys].avg)
                    if comet is not None:
                        comet.log_metric('Val Adv NOT Attacked Task/ Avg clean Loss {}'.format(keys), avg_losses[keys].avg)
                        comet.log_metric('Val Adv NOT Attacked Task/ Avg clean MAE {}'.format(keys), avg_errors[keys].avg)
                        comet.log_metric('Val Adv NOT Attacked Task/ Avg clean MSE {}'.format(keys), avg_mse[keys].avg)
                        comet.log_metric('Val Adv NOT Attacked Task/ Avg adv Loss {}'.format(keys),
                                         adv_avg_losses[keys].avg)
                        comet.log_metric('Val Adv NOT Attacked Task/ Avg adv MAE {}'.format(keys), adv_avg_errors[keys].avg)
                        comet.log_metric('Val Adv NOT Attacked Task/ Avg adv MSE {}'.format(keys), adv_avg_mse[keys].avg)

        if 'segmentsemantic' in criterion.keys() or 'segmentsemantic' in criterion.keys():
            ious = per_class_iu(hist) * 100
            logger.info(' '.join('{:.03f}'.format(i) for i in ious))
            mIoU = round(np.nanmean(ious), 2)

            adv_ious = per_class_iu(adv_hist) * 100
            adv_mIoU = round(np.nanmean(adv_ious), 2)

            str_attack_result += '\n Segment Score ({score.avg:.3f}) \t'.format(score=seg_accuracy)
            str_attack_result += ' Segment ===> mAP {}\n'.format(mIoU)

            if comet is not None:
                comet.log_metric('segmentsemantic Original IOU', mIoU)
                comet.log_metric('segmentsemantic Attacked IOU',    adv_mIoU)
                comet.log_metric('segmentsemantic Attacked Score',  seg_accuracy.avg)

    if test_flag:

        dict_losses = {}
        for key, loss_term in criterion.items():
            dict_losses[key] = avg_losses[key].avg
            # print(str_attack_result, "\nnew", avg_losses[keys].avg, "\n")
        if 'segmentsemantic' in criterion.keys():
            dict_losses['segmentsemantic'] = {'iou'    : mIoU,
                                              'loss'   : avg_losses['segmentsemantic'].avg,
                                              'seg_acc': seg_accuracy.avg}

        print("These losses are returned", dict_losses)
        #Compute the dictionary of losses that we want. Desired: {'segmentsemantic:[mIoU, cel],'keypoints2d':acc,'}
        return dict_losses, step_metrics

    return None

