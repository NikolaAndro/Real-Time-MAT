import torch
from MAT_utils.utils_MAT_attack_minimal import fitness_evaluation_validation, normalize_and_scale
from MAT_utils.utils_aux_MAT_attack import AverageMeter
import time 

def validate_model(MATAttackModel, mtask_model, val_loader, criteria, A, b, mean, std, args, Experiment = None ):
    '''Validates the model based on the test batch from validation loader.'''
    # average of each loss over epochs
    avg_losses = {}
    adv_avg_losses = {}

    for targeted_task, task_loss_fn in criteria.items():
        avg_losses[targeted_task] = AverageMeter()
        avg_losses[targeted_task+"_list"] = None
        adv_avg_losses[targeted_task] = AverageMeter()
        adv_avg_losses[targeted_task+"_list"] = None


    # Follow fitness over epochs so we can plot
    clean_fitness_batch_avg = AverageMeter()
    adv_fitness_batch_avg = AverageMeter()

    mtask_model.eval()
    MATAttackModel.eval()

    with torch.no_grad():
        for i, (test_input_batch, test_target_batch, test_mask_batch, test_file_name) in enumerate(val_loader):
            # send data to cuda if available
            if torch.cuda.is_available():
                if test_input_batch.is_cuda == False:
                    test_input_batch = test_input_batch.cuda()
                for keys, tar in test_target_batch.items():
                    if tar.is_cuda == False:
                        test_target_batch[keys] = tar.cuda()
                for keys, m in test_mask_batch.items():
                    if m.is_cuda == False:
                        test_mask_batch[keys] = m.cuda()

            # convert target batch to float
            if type(criteria) in [torch.nn.modules.loss.L1Loss, torch.nn.modules.loss.MSELoss]:
                test_target_batch = test_target_batch.float()

            if args.inference_time: inference_time_start = time.time()

            batch_size = args.batch_size if args.mode == "train" else args.test_batch_size
            perturbation = MATAttackModel(test_input_batch) 
            perturbation = normalize_and_scale(perturbation, mean_arr=mean, stddev_arr=std, batch_size = batch_size)
            perturbed_images = torch.add(test_input_batch, perturbation.cuda(0))
            # do clamping per channel
            for channel in range(3):
                perturbed_images[:,channel,:,:] = perturbed_images[:,channel,:,:].clone().clamp(test_input_batch[:,channel,:,:].min(), test_input_batch[:,channel,:,:].max())

            if args.inference_time: print("\n\nInference time of the MAT-Attack is: ", time.time() - inference_time_start," seconds.\n\n")

            #reduce memory consumption for computations that would otherwise have requires_grad=True
            clean_output = mtask_model.forward(test_input_batch)
            adversarial_output = mtask_model.forward(perturbed_images)

            clean_losses = {}
            adv_losses = {}

            for targeted_task, task_loss_fn in criteria.items():
                # clean loss
                clean_loss = task_loss_fn(clean_output[targeted_task].float(), test_target_batch[targeted_task], \
                                                test_mask_batch[targeted_task]).item()
                clean_losses[targeted_task] = clean_loss
                avg_losses[targeted_task].update(clean_loss, args.batch_size)

                # adversarial loss
                adv_loss = task_loss_fn(adversarial_output[targeted_task].float(), test_target_batch[targeted_task],\
                                                    test_mask_batch[targeted_task]).item()
                adv_losses[targeted_task] = adv_loss
                adv_avg_losses[targeted_task].update(adv_loss, args.batch_size)

            # free cache memory
            del adversarial_output, clean_output

            torch.cuda.empty_cache()

            # clean fitness
            clean_fitness_batch_avg.update(fitness_evaluation_validation(clean_losses,A,b,args))

            # adversarial fitness
            adv_fitness_batch_avg.update(fitness_evaluation_validation(adv_losses,A,b,args))

            if args.single_img_tests:
                break

    # keep a list of average test losses
    for targeted_task, task_loss_fn in criteria.items():
        avg_losses[targeted_task+"_list"]=avg_losses[targeted_task].avg
        adv_avg_losses[targeted_task+"_list"]=adv_avg_losses[targeted_task].avg

    if Experiment != None:
        # keep a list of average test losses
        for targeted_task, task_loss_fn in criteria.items():
            Experiment.log_metric("test_clean_loss_"+targeted_task, avg_losses[targeted_task].avg)
            Experiment.log_metric("test_adv_loss_"+targeted_task, adv_avg_losses[targeted_task].avg)
            
        # keep fitness lists updated
        Experiment.log_metric("test_clean_fitness", clean_fitness_batch_avg.avg)
        Experiment.log_metric("test_adv_fitness", adv_fitness_batch_avg.avg)
     
    # build final dict that contains all necessary info and return it
    info = {}
    info['test_clean_loss']             = avg_losses
    info['test_adv_loss']               = adv_avg_losses
    info['test_clean_fitness']          = clean_fitness_batch_avg.avg
    info['test_adv_fitness']            = adv_fitness_batch_avg.avg

    return info
  
                
