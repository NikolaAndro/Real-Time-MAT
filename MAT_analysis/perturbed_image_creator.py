import datetime
import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from learning.dataloader import  get_loader
import datetime
import config
import MAT_utils.utils_aux_MAT_attack as MAT_AUX
import MAT_utils.utils_MAT_attack_minimal as MAT
from learning.MAT_learning.model_loading import *
from mtask_models.mtask_losses import get_losses_and_tasks
import torchvision.utils as utils

parser = argparse.ArgumentParser(description='Run Adversarial attacks experiments')
parser.add_argument('--arch_tau', type=str, default="resnet")
parser.add_argument('--arch', type=str, default="xception-full")
parser.add_argument('--using_iterations', type=MAT_AUX.str2bool, default=True)
parser.add_argument('--dataset', type=str, default="taskonomy")
parser.add_argument('--data_dir', type=str, default=config.TASKONOMY_DATASET)
parser.add_argument('--workers',type=int, default=8)
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer: for now, only "SGD"')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--random_seed',type=int, default=3906)
parser.add_argument("--save_model", type=MAT_AUX.str2bool, nargs='?', const=True, default=False, help="Saving models or not.")
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--norm_type', type=str, default="batch", help='norm_type: "instance" or "batch"')
parser.add_argument('--act_type', type=str, default="relu", help='activation function: "relu" or "selu"')
parser.add_argument('--gpu_ids', help='gpu ids: e.g. 0 or 0,1 or 1,2.', type=str, default='0')
parser.add_argument('--num_init_models',type=int, default=10, help="number of initial models for training process")
parser.add_argument('--pretrained_mtask_model_path',type=str, default=config.PRETRAINED_MTASK_MODEL_PATH)
parser.add_argument('--pretrained_generator_model_path',type=str, default="/home/namdd/Documents/github_NOT_ALL_backed_up/paper_results/code_v2/2_POF_2_predators/23G/results/2023-4-26___11:23:27/checkpoint_6.pth.tar")
parser.add_argument('--experiment_backup_folder',type=str, default=config.BACKUP_OUTPUT_DIR)
parser.add_argument('--debug_mode',type=MAT_AUX.str2bool,nargs='?', const=True, default="False", help="Enables print statements.")
parser.add_argument('--gpu_debug',type=MAT_AUX.str2bool,nargs='?', const=True, default="False", help="Enables gpu debug comet logs.")
parser.add_argument("--comet", type=MAT_AUX.str2bool, nargs='?', const=True, default=False, help="Report data to comet online.")
parser.add_argument("--include_mutation", type=MAT_AUX.str2bool, nargs='?', const=True, default=True, help="")
parser.add_argument('--num_carryovers',type=int, default=10, help="number of tau models that will be passed into the next iteration.")
parser.add_argument('--save_on_every',type=int, default=1, help="Save the model on every nth iteration.")
parser.add_argument('--POF_type', type=str, default="nonlinear", help='POF type: "linear" or "nonlinear"')
parser.add_argument('--mode', type=str, default="train", help='"train" or "test"')
parser.add_argument('--dropout', type=MAT_AUX.str2bool, default=False)
parser.add_argument('--inference_time', type=MAT_AUX.str2bool, default=False)
parser.add_argument('--activ_func', type=str, default="relu", help='"relu" or "selu"')
parser.add_argument('--single_color', type=MAT_AUX.str2bool, default=False, help='Using only one color on the plots.')


parser.add_argument('--num_epochs',type=int, default=1)
parser.add_argument('--batch_size',type=int, default=1)
parser.add_argument('--test_batch_size',type=int, default=1)
parser.add_argument('--train_task_set', default="dn")
parser.add_argument('--targeted_tasks_set', default="dn")
parser.add_argument('--crossover_vers', help='e.g. 1 or 2.', type=str, default='2')
parser.add_argument('--crossover_dim', help='e.g. 0 or 1 or 2 or 3.', type=str, default='3')
parser.add_argument("--greedy_approach", type=MAT_AUX.str2bool, nargs='?', const=True, default=False, help="Usinig greedy approach or not.")
parser.add_argument('--num_POF_points',type=int, default=2, help="Number of points on pareto-optimal front.")
parser.add_argument('--num_grad_predators',type=int, default=5, help="number of tau models that closest to eachPOF point in gradient computation.")
parser.add_argument("--save_plots", type=MAT_AUX.str2bool, nargs='?', const=True, default=False, help="Creating plots or not.")
parser.add_argument('--single_img_tests', type=MAT_AUX.str2bool, default=False)
parser.add_argument('--num_loop_iters',type=int, default=1, help="Number of loop iterations for single image tests.")
parser.add_argument('--num_reproduction_models',type=int, default=2, help="Number of models to be generated in reproduction phase.")
parser.add_argument('--mutation_multiple',type=int, default=2, help="Number of layers to mutate in each mutation.")
parser.add_argument('--num_mutations',type=int, default=4, help="Number of models to be generated in mutation phase.")
parser.add_argument('--p_lambda', type=float, default=1.0, help='Lambda for logit probability computation.')

args = parser.parse_args()
gpulist = [int(i) for i in args.gpu_ids.split(',')]

# Tiny Taskonomy 
mean = [-0.0648, -0.0785, -0.0902]
std = [0.4412, 0.4594, 0.4837]

def generate_images(MATAttackModel, mtask_model, val_loader):
    '''Saves the input image and its perturbation.'''
    mtask_model.eval()
    MATAttackModel.eval()
    
    with torch.no_grad():
        for test_input_batch, __, _, test_file_name in val_loader:
            print(test_file_name)

            # send data to cuda if available
            if torch.cuda.is_available():
                if test_input_batch.is_cuda == False:
                    test_input_batch = test_input_batch.cuda()
                
            nonscaled_perturbedation = MATAttackModel(test_input_batch)

            batch_size = args.test_batch_size
            perturbedation = MAT.normalize_and_scale(nonscaled_perturbedation, mean_arr=mean, stddev_arr=std, batch_size = batch_size)
            
            perturbed_image = torch.add(test_input_batch, perturbedation.cuda(0))
            # do clamping per channel
            for channel in range(3):
                perturbed_image[:,channel,:,:] = perturbed_image[:,channel,:,:].clone().clamp(test_input_batch[:,channel,:,:].min(), test_input_batch[:,channel,:,:].max())
    
            utils.save_image(test_input_batch[21], 'images/final_models/clean_image.png')
            utils.save_image(perturbed_image[21], 'images/final_models/adversarial_image.png')
            utils.save_image(perturbedation[21], 'images/final_models/perturbedation.png')
            utils.save_image(nonscaled_perturbedation[21], 'images/final_models/nonscaled_perturbedation.png')
            break
    return 
  
def main(args):
    '''Trains a MAT-Attack model using the given parameters.'''
    args.task_set = MAT_AUX.abbrev_to_task(args.train_task_set)

    # mean, std = MAT_AUX.dataset_mean_and_std(args, treshold=1200)

    # print(mean)
    # print(std)

    if args.arch_tau == 'resnet':
        # model is automatically on CUDA
        model = ResnetGenerator(3, 3, args.ngf, norm_type=args.norm_type, act_type=args.activ_func, gpu_ids=gpulist, n_blocks=6, use_dropout=args.dropout)    

        # model.apply(weights_init)
        # In case of pretrained model
        # if args.debug_mode: print("\n\nLoading a pretrained GENERATOR model...\nPretrained model path: ",args.pretrained_generator_model_path)
        model.load_state_dict(torch.load(args.pretrained_generator_model_path))   

        model = torch.nn.DataParallel(model)
        model.eval()
    else:
        raise Exception("Only resnet architecture is implemented so far.")


    ################################# Load mtask model #################################
    mtask_model = load_mtask_model(args)

    # MAT_AUX.seed_everything(args.random_seed)

    args.test_batch_size = 64
    val_loader = get_loader(args, "val", treshold=500)
   
    #***************************************************************************
    #********************************* Images **********************************
    #***************************************************************************
    generate_images(model.module, mtask_model, val_loader)

    print("Images are saved.")

if __name__ == '__main__':
    main(args)

