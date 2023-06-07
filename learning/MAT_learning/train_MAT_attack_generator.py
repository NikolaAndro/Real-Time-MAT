import datetime
import os
import time
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from learning.dataloader import  get_loader
import datetime
import config
import MAT_utils.utils_MAT_attack_minimal as MAT
import MAT_utils.utils_aux_MAT_attack as MAT_AUX
from learning.MAT_learning.GAMAT_training_loop import GAMAT
import learning.MAT_learning.MAT_validation as MAT_validate
from learning.MAT_learning.model_loading import *
from tqdm import tqdm
from mtask_models.mtask_losses import get_losses_and_tasks

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
parser.add_argument('--pretrained_generator_model_path',type=str, default="")
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

# The Pareto-Optimal Front must be given as a list of points on imaginary line.
# NOTE: you can use this tool to find the points - https://www.desmos.com/calculator/zzxbryrahc
if args.train_task_set == 'dn' :
    if args.POF_type == "linear":
        # y = (2-x)/1.75
        A = np.array([[1,1.75]])
        b = np.array([2.0])
        # lambda = (y = (2-x)/1.75) --> linear POF this is not code
        if args.num_POF_points == 5:
            POF = [[0.4 , 0.91428571],
                [0.75, 0.71428571],
                [1.075,0.52857143],
                [1.4 , 0.34285714],
                [1.7 , 0.17142857]]   
        elif args.num_POF_points == 2:
            POF = [[0.75, 0.71428571],
                    [1.4 , 0.34285714]] 
        POF_to_plot = lambda pof_x: (2 - pof_x)/1.75
        x_axis_width = [0.000001,2] # how much of X axis will be in the graph
        y_axis_height = 1.4 

    elif args.POF_type == "nonlinear":
        # y = 1 / 1.8x  nonlinear POF
        A = np.array([[1.8,1.0]])
        b = np.array([1.0]) 
        if args.num_POF_points == 5:
            POF = [[0.7   , 0.79365079],
                [1.05  , 0.52910053],
                [1.45  , 0.38314176],
                [1.9   , 0.29239766],
                [2.5   , 0.22222222]]   
        elif args.num_POF_points == 2:
            POF = [[0.7 , 0.79365079],
                   [2.5, 0.222222222]]
        elif args.num_POF_points == 3:
            POF = [[0.7 , 0.79365079],
                   [1.45  , 0.38314176],
                   [2.5, 0.222222222]]

        POF_to_plot = lambda pof_x: 1 / (1.8 * pof_x)
        x_axis_width = [0.000001, 4]
        y_axis_height = 1.4

        # y = 2 / x  nonlinear POF
        # A = np.array([[0.5  ,1.0]])
        # b = np.array([1.0]) 
        # if args.num_POF_points == 5:
        #     POF = [[1.33 , 1.5037594],
        #            [1.78 , 1.1235955],
        #            [2.3  , 0.86956522],
        #            [2.9  , 0.68965517],
        #            [3.5  , 0.57142857]]   
        # elif args.num_POF_points == 2:
        #     POF = [[1.33 , 1.5037594],
        #            [3.5  , 0.57142857]]
        # elif args.num_POF_points == 3:
        #     POF = [[1.33 , 1.5037594],
        #            [2.3  , 0.86956522],
        #            [3.5  , 0.57142857]]

        # POF_to_plot = lambda pof_x: 2 / pof_x
        # x_axis_width = [0.000001, 4]
        # y_axis_height = 2.2
else:
    raise Exception("The POF for this combination of tasks is not set up. Please do so.")     

crossover_versions = [int(i) for i in args.crossover_vers.split(',')]
crossover_dimensions = [int(i) for i in args.crossover_dim.split(',')]

# GPU usage
gpu_y_axis_percentage = []
gpu_y_axis_num_obj = []

# The Best Fitness over epochs
epochs = np.arange(args.num_epochs, dtype=int)
best_fitness_y_axis = []

# These need to be calculated as follows:
# mean, std = dataset_mean_and_std(args)
# Tiny Taskonomy 
mean = [-0.0648, -0.0785, -0.0902]
std = [0.4412, 0.4594, 0.4837]

# save the results in a file
t = datetime.datetime.now()
time_stamp = str(t.year)+"-"+str(t.month)+"-"+str(t.day)+"___"+str(t.hour)+":"+str(t.minute)+":"+str(t.second)
today = str(t.year)+"-"+str(t.month)+"-"+str(t.day)+"___"+str(t.hour)+":"+str(t.minute)+":"+str(t.second)+'/'
args.experiment_backup_folder = args.experiment_backup_folder+today
os.makedirs(args.experiment_backup_folder, exist_ok=True)
os.makedirs(args.experiment_backup_folder+"/iterations/", exist_ok=True)
os.makedirs(args.experiment_backup_folder+"/iterations/cross/", exist_ok=True)
final_model_path = os.path.join(args.experiment_backup_folder, 'final_model_on_' + time_stamp + '.pth.tar')

good_init_check = False

def train_MAT_attak(args):
    '''Trains a MAT-Attack model using the given parameters.'''
    global A, b
    checkpoint_lists_file = open(args.experiment_backup_folder+"result_lists_iterations.txt", "a")

    if args.comet:
        # Comet chart
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        if args.greedy_approach == False:
            experiment_name = args.train_task_set + "_" + str(args.num_POF_points) +"pts_" + str(args.num_grad_predators) + "predators_" + \
                                str(args.crossover_vers)+ "_"+ str(args.crossover_dim) + "_" + args.dataset + "_" + timestamp 
        else:
            experiment_name = args.train_task_set + "_" + str(args.num_POF_points) +"pts_" + str(args.num_grad_predators) + "predators_" + \
                                str(args.crossover_vers)+ "_"+ str(args.crossover_dim) + "_G_" + args.dataset + "_" + timestamp 
        experiment = MAT_AUX.init_comet(args,project_name="MAT-Attack")
        experiment.set_name(experiment_name)
        experiment.log_parameters(vars(args))
    else:
        experiment = None

    # Load the data
    args.task_set = MAT_AUX.abbrev_to_task(args.train_task_set)
    args.targeted_tasks_set = MAT_AUX.abbrev_to_task(args.targeted_tasks_set)
   
    criteria, targeted_tasks = get_losses_and_tasks(args)

    if args.debug_mode: print("\nAttacking the following tasks: ", targeted_tasks)

    ################################# Load initial tau models with random weights #################################
    initial_tau_models = load_generators(crossover_versions,crossover_dimensions, config, args, gpulist, experiment)

    ################################# Load mtask model #################################
    mtask_model = load_mtask_model(args)

    ################################# Load  training data and optimizer #################################
    # Set the same seed every time to repeat the same scenario if desired
    MAT_AUX.seed_everything(args.random_seed)

    train_loader = get_loader(args, "train")
    val_loader = get_loader(args, "val")

    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
    # reulsts in a faster running time
    cudnn.benchmark = True

    ################################# Generate the best model #################################
    top_model = None
    
    if args.debug_mode:
        MAT_AUX.dance_baby()
        print("Dancing....")
    
    train_fitness_avg_list_over_epochs = []
    train_adv_avg_losses = {}

    # *************** Validation Info Tracking Lists ****************
    test_avg_losses = {}
    test_adv_avg_losses = {}

    for targeted_task in criteria.keys():
        test_avg_losses[targeted_task+"_list"] = []
        test_adv_avg_losses[targeted_task+"_list"] = []
        train_adv_avg_losses[targeted_task] = MAT_AUX.AverageMeter()
        train_adv_avg_losses[targeted_task+"_list"] = []

    # Follow fitness over epochs so we can plot
    test_clean_fitness_list = []
    test_adv_fitness_list = []

    for epoch in tqdm(range(args.num_epochs), desc="epoch"):
        if args.comet: experiment.log_metric("MAT_epoch",epoch)
        # Track the averages
        train_fitness_avg_over_batches = MAT_AUX.AverageMeter()

        for targeted_task in criteria.keys():
            train_adv_avg_losses[targeted_task] = MAT_AUX.AverageMeter()

        #***************************************************************************
        #******************************** Training *********************************
        #***************************************************************************
        for i, (input_batch, target_batch, mask_batch, file_name) in enumerate(train_loader):  
            # convert target batch to float
            if type(criteria) in [torch.nn.modules.loss.L1Loss, torch.nn.modules.loss.MSELoss]:
                target_batch = target_batch.float()

            new_top_models = GAMAT(input_batch, target_batch, mask_batch, initial_tau_models,
                                    mtask_model, criteria, targeted_tasks, epoch,args, mean, std, POF, A, b, 
                                    gpulist, crossover_versions, crossover_dimensions, good_init_check, 
                                    POF_to_plot, x_axis_width, y_axis_height, experiment = experiment)                 

            initial_tau_models = new_top_models
            
            # Top model
            if epoch % args.save_on_every == 0 and args.save_model or \
                i == len(train_loader)-1 and epoch == args.num_epochs-1 or\
                args.single_img_tests:
                top_model = MAT.model_manual_copy(new_top_models[0],args, gpulist)

            # average of the training fitness over the best models of each batch
            train_fitness_avg_over_batches.update(new_top_models[0].final_fitness_distance)

            # keep the average of losses over batches to get the average losses of epoch
            for targeted_task in criteria.keys():
                train_adv_avg_losses[targeted_task].update(new_top_models[0].empirical_losses[targeted_task], args.batch_size)

            if args.single_img_tests:
                epochs = np.arange(epoch+1)
                break
        
        #********************* END OF AN EPOCH *******************
        # Update y axis for plt image
        best_fitness_y_axis.append(new_top_models[0].final_fitness_distance)
        
        # Measure the average fitness for epoch
        train_fitness_avg_list_over_epochs.append(train_fitness_avg_over_batches.avg)
        for targeted_task in criteria.keys():
            train_adv_avg_losses[targeted_task+"_list"].append(train_adv_avg_losses[targeted_task].avg)

        if args.comet:
            experiment.log_metric("train_fitness_average_epoch",train_fitness_avg_over_batches.avg)
            experiment.log_metric("epoch_top_model_fitness", new_top_models[0].final_fitness_distance)
            experiment.log_metric("epoch", epoch)
            for targeted_task in criteria.keys():
                experiment.log_metric("training_epoch_avg_loss_"+targeted_task, train_adv_avg_losses[targeted_task].avg)

        # save checkpoints
        if epoch % args.save_on_every == 0 and args.save_model:
            checkpoint_path = os.path.join(args.experiment_backup_folder, 'checkpoint_'+str(epoch)+'.pth.tar')
            print("\n\nSaving checkpoint on epoch "+str(epoch)+" to " + checkpoint_path +" ....\n")
            torch.save(top_model.model.module.state_dict(), checkpoint_path)
                
        MAT_AUX.save_lists_in_checkpoint('train', epoch, checkpoint_lists_file, criteria, \
            train_fitness_avg_list_over_epochs=train_fitness_avg_list_over_epochs, \
                train_adv_avg_losses= train_adv_avg_losses)

        #***************************************************************************
        #******************************* Validation ********************************
        #***************************************************************************
        validation_info = MAT_validate.validate_model(top_model.model.module, mtask_model, val_loader, criteria, A, b, mean, std, args, Experiment=experiment)

        # Update test lists
        for targeted_task in criteria.keys():
            test_avg_losses[targeted_task+"_list"].append(validation_info['test_clean_loss'][targeted_task].avg)
            test_adv_avg_losses[targeted_task+"_list"].append(validation_info['test_adv_loss'][targeted_task].avg)
        
        # Follow fitness over epochs so we can plot
        test_clean_fitness_list.append(validation_info['test_clean_fitness'])
        test_adv_fitness_list.append(validation_info['test_adv_fitness'])

        # Save the log info
        MAT_AUX.save_lists_in_checkpoint('validate', epoch, checkpoint_lists_file, criteria, \
            test_clean_fitness_list = test_clean_fitness_list,test_adv_fitness_list=test_adv_fitness_list,\
            test_avg_losses=test_avg_losses,test_adv_avg_losses=test_adv_avg_losses)

        # If the POF is reached, the algorithm converged
        if train_fitness_avg_over_batches.val <= 0:
            print("\n\nThe algorithm has converged.\n\n")
            break

    if args.debug_mode:
        MAT_AUX.dance_finished()
    
    if args.save_model:
        print("Saving final model to " + final_model_path+" ....")
        torch.save(top_model.model.module.state_dict(), final_model_path)

    #save characteristics
    MAT_AUX.save_characteristics(top_model, train_fitness_avg_list_over_epochs, train_adv_avg_losses, test_clean_fitness_list,\
                        test_adv_fitness_list, test_avg_losses, test_adv_avg_losses, criteria, args)

    MAT_AUX.plot_graph(epochs, best_fitness_y_axis, args, experiment, 'g', 'train_fithess', "Epoch", "The Best Fitness", "Model Fitness Over Epochs", "model_fitness.pdf", 3)
   
if __name__ == '__main__':
    train_MAT_attak(args)

