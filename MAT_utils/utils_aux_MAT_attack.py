import matplotlib.pyplot as plt 
import subprocess
import os
import argparse
import torch
import numpy as np
import random
import psutil
import emoji
from colorama import init, Fore, Back, Style
from comet_ml import Experiment
import config
from learning.dataloader import  get_loader

map_dict = {
        'segmentsemantic'    : 's',
        'edge_texture'       : 'e',
        'depth_zbuffer'      : 'd',
        'autoencoder'        : 'A',
        'edge_occlusion'     : 'E',
        'depth_euclidean'    : 'D',
        'normal'             : 'n',
        'principal_curvature': 'p',
        'reshading'          : 'r',
        'keypoints2d'        : 'k',
        'keypoints3d'        : 'K'
    }

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def str2bool(v):
    """Accepts multiple values for boolean in terminal."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_losses(label, omega, targeted_tasks_set, plot_color, directory, y_axis_height, args):
    tasks = []
    for letter in targeted_tasks_set:
        tasks.append(abbrev_to_task(letter)[0])
    #plotting for testing
    cross_l1 =[]
    cross_l2 =[]
    
    for tau in omega:
        counter = 1
        for task_name in tasks:
            if counter == 1:
                if isinstance(tau.empirical_losses[task_name],float):
                    cross_l1.append(tau.empirical_losses[task_name])
                else:
                    cross_l1.append(tau.empirical_losses[task_name].item())
                counter += 1
            else:
                if isinstance(tau.empirical_losses[task_name],float):
                    cross_l2.append(tau.empirical_losses[task_name])
                else:
                    cross_l2.append(tau.empirical_losses[task_name].item())
                counter = 1

    plt.xlabel(tasks[0])
    plt.ylabel(tasks[1])
    plt.ylim(0,y_axis_height)
    plt.plot(cross_l1,cross_l2, color=plot_color,label=label, marker='o', ls = '')

    if not args.single_color:
        # get handles and labels for all plots
        handles, labels = plt.gca().get_legend_handles_labels()

        # update legend with new handles and labels
        plt.legend(handles, labels)

    plt.savefig(directory)

def plot_POF(y_function, POF,x_axis_width):
    # 100 linearly spaced numbers
    pof_x = np.linspace(x_axis_width[0],x_axis_width[1],100)
    pof_y = y_function(pof_x)#(2 - pof_x)/1.8

    # plot the function
    plt.plot(pof_x,pof_y, 'g', label='POF')    
    xx = []
    yy = []
    for point in POF:
        xx.append(point[0])
        yy.append(point[1])

    plt.plot(xx,yy, color = 'b', marker='o', label='POF_points', ls = '')
    plt.legend()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def find_min_fitness_distance(omega):
    # NOTE: Using min function above is slower than the below method.
    top_fitness_after = 9999999
    for i in range(len(omega)):
        if omega[i].final_fitness_distance < top_fitness_after:
            top_fitness_after = omega[i].final_fitness_distance
    return top_fitness_after

def dance_baby():
    # emoji:  https://unicode.org/emoji/charts/full-emoji-list.html
    # Initializes Colorama: https://stackabuse.com/how-to-print-colored-text-in-python/
    init(autoreset=True) 
    print("\t\t     ",emoji.emojize(":woman_dancing:"),emoji.emojize(":man_dancing:"),emoji.emojize(":woman_dancing:"),\
        emoji.emojize(":man_dancing:"),emoji.emojize(":woman_dancing:"),emoji.emojize(":man_dancing:"),"\n")
    print(Style.BRIGHT + Back.CYAN + Fore.RED + "                                                             ")
    print(Style.BRIGHT + Back.CYAN + Fore.RED + "     -----------     LET THE DANCE BEGIN      ----------     ")
    print(Style.BRIGHT + Back.CYAN + Fore.RED + "                                                             ")
    print("\n\t\t\t ",emoji.emojize(":winking_face:"),emoji.emojize(":winking_face:"),emoji.emojize(":winking_face:"),"\n")   

def dance_finished():
    init(autoreset=True) 
    print("\n\t\t     ",emoji.emojize(":woman_dancing:"),emoji.emojize(":man_dancing:"),emoji.emojize(":woman_dancing:"),\
        emoji.emojize(":man_dancing:"),emoji.emojize(":woman_dancing:"),emoji.emojize(":man_dancing:"),"\n")
    print(Style.BRIGHT + Back.CYAN + Fore.RED + "                                                             ")
    print(Style.BRIGHT + Back.CYAN + Fore.RED + "      -----------      DANCE FINISHED       ----------       ")
    print(Style.BRIGHT + Back.CYAN + Fore.RED + "                                                             ")
    print("\n\t\t      ",emoji.emojize(":collision:"),emoji.emojize(":hundred_points:"),emoji.emojize(":hundred_points:"),\
        emoji.emojize(":hundred_points:"),emoji.emojize(":collision:"),"\n")   


# ************************************ Memory Helpers ****************************************
def show_gpu(msg, printer = True):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    def query(field):
        return(subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
                '--format=csv,nounits,noheader'], 
            encoding='utf-8'))
    def to_int(result):
        return int(result.strip().split('\n')[0])
    
    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used/total
    if printer == True:
        print('\n' + msg, f'{100*pct:2.1f}% ({used}MiB out of {total}MiB)') 
    return pct*100

def get_RAM_used_perc():
    return psutil.virtual_memory()[2]

def print_gpu_obj(just_num_objects=True, printer = True):
    '''
        Desctiprion:
            - Prints number of objects on GPU.
    '''
    import gc
    count = 0 
    for tracked_object in gc.get_objects():
        if torch.is_tensor(tracked_object):
            if tracked_object.is_cuda:
                count+=1
                if just_num_objects == False:
                    if printer == True:
                        print("{} {} {}".format(
                            type(tracked_object).__name__,
                        " pinned" if tracked_object.is_pinned() else "",
                            tracked_object.shape
                            ))
    if printer == True:
        print(f'\nTHERE ARE {count} OBJECTS ON GPU')
    return count

def save_lists_in_checkpoint(mode, epoch, checkpoint_lists_file,criteria, \
                            train_fitness_avg_list_over_epochs=None, \
                            train_adv_avg_losses=None, \
                            test_clean_fitness_list=None,\
                            test_adv_fitness_list=None, \
                            test_avg_losses=None ,\
                            test_adv_avg_losses=None):
    '''
        Desctiption:
            - Saves information about training to enable us to reconstruct graphs at any point during the training.
        Arguments: 
            - mode: (string) 'train' or 'validate'.
            - epoch: current epoch index
            - checkpoint_lists_file: file to write results to.
            - if mode=='train': 
                - train_fitness_avg_list_over_epochs
                - train_adv_avg_losses
            - if mode=='validate':
                - test_clean_fitness_list
                - test_adv_fitness_list
                - test_avg_losses
                - test_adv_avg_losses
            - Lists for tracking the training and testing info.
        Returns: Nothing. It writes this info to the given file.
    '''
    checkpoint_lists_file.write("\n******************* EPOCH " + str(epoch) + " " + mode + " *******************\n\n")
    if mode == 'train':  
        checkpoint_lists_file.write("train_fitness_avg_list_over_epochs: " + str(train_fitness_avg_list_over_epochs)+"\n")
        for targeted_task, task_loss_fn in criteria.items():
            checkpoint_lists_file.write("train_epoch_loss_avg_"+targeted_task+": " + str(train_adv_avg_losses[targeted_task+"_list"])+"\n")
    elif mode == 'validate':
        # VALIDATE  info
        checkpoint_lists_file.write("test_clean_fitness_list_over_epochs"  +": " +str(test_clean_fitness_list)+"\n")
        checkpoint_lists_file.write("test_adv_fitness_list_over_epochs"  +": " +str(test_adv_fitness_list)+"\n")
       
        for targeted_task, task_loss_fn in criteria.items():
            checkpoint_lists_file.write("test_clean_loss_"+targeted_task+": " + str(test_avg_losses[targeted_task+"_list"])+"\n")
            checkpoint_lists_file.write("test_adv_loss_"+targeted_task+": " + str(test_adv_avg_losses[targeted_task+"_list"])+"\n")
    else:
        raise Exception("This is an unknown mode. Valid optinos: 'train' and 'validate'.")
    
    checkpoint_lists_file.flush()

def save_characteristics(top_model, train_fitness_avg_list_over_epochs, train_adv_avg_losses, test_clean_fitness_list,\
                        test_adv_fitness_list, test_avg_losses, test_adv_avg_losses, criteria, args):
    ''''
    Description: 
        - Saves characteristics of a model
    '''
    print("Saving top model characteristics to " + args.experiment_backup_folder+"model_characteristics.txt"+" ....")
    characteristics_file = open(args.experiment_backup_folder+"model_characteristics.txt", "w")
    for key, val in top_model.model_characteristics.items():
        if key != 'initialization_method(s)':
            characteristics_file.write(str(key) + ": " + str(val) + "\n")
        else:
            characteristics_file.write(str(key) + ": [\n")
            for i in range(len(val)):
                if i != len(val)-1:
                    characteristics_file.write("\t\t\t   "+ str(val[i]) + ",\n")
                else:
                    characteristics_file.write("\t\t\t   "+ str(val[i]) + "    ]")
            characteristics_file.write("\n\n\n")
        
    characteristics_file.write("train_fitness_avg_list_over_epochs: " + str(train_fitness_avg_list_over_epochs)+"\n")
    # VALIDATE  info
    characteristics_file.write("test_clean_fitness_list_over_epochs: " + str(test_clean_fitness_list)+"\n")
    characteristics_file.write("test_adv_fitness_list_over_epochs: " + str(test_adv_fitness_list)+"\n\n\n")

    for targeted_task in criteria.keys():
        characteristics_file.write("train_epoch_loss_avg_"+targeted_task+": " + str(train_adv_avg_losses[targeted_task+"_list"])+"\n")
        characteristics_file.write("test_clean_loss_"+targeted_task+": " + str(test_avg_losses[targeted_task+"_list"])+"\n")
        characteristics_file.write("test_adv_loss_"+targeted_task+": " + str(test_adv_avg_losses[targeted_task+"_list"])+"\n")

    characteristics_file.close()

def abbrev_to_task(abbrev):
    ''''
    Description: 
        - Converts task abbreviations to full task names.
    Arguments: 
        - abbrev: (list) abbreviations for a tasks
    Returns:
        - list of fully named tasks.
    '''
    if not abbrev:
        return abbrev
    inverse_dict = dict(zip(map_dict.values(), map_dict.keys()))
    tasks = [inverse_dict.get(e,None) for e in abbrev if inverse_dict.get(e,None)]
    return tasks

def init_comet(args, project_name="MAT-attack"):
    ''''
    Description: 
        - Initiates teh comet project that enables tracking/logging of information.
    Arguments: 
        - args
        - project_name: (str) name for the project in comet workspace.
    Returns:
        - comet experiment object. 
    '''
    experiment = Experiment(api_key=config.COMET_APIKEY,
                            project_name=project_name,
                            workspace="",
                            auto_param_logging=False, auto_metric_logging=False,
                            parse_args=False, display_summary=False, disabled=False)
    experiment.log_parameters(vars(args))
    return experiment

def plot_graph(x, y, args, experiment, colorr, labela, x_label, y_label, title, img_name, figure_index ):
    b = plt.figure(figure_index)
    plt.plot(x,y, color = colorr, marker='o', label=labela)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(args.experiment_backup_folder+img_name)
    # send it to comet
    if args.comet:
        experiment.log_image(args.experiment_backup_folder+img_name)

def dataset_mean_and_std(args, treshold=9999999999999):
    '''Calculates the mean and the standard deviation of the dataset'''
    channels_sum, channels_squared_sum, num_batches = 0,0,0

    loader = get_loader(args, "train", treshold=treshold)
    for input_batch, labels, masks, file_name in loader:
        channels_sum += torch.mean(input_batch, dim=[0,2,3]) 
        channels_squared_sum += torch.mean(input_batch**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    print(type(mean))
    print(type(std))
    return mean, std
