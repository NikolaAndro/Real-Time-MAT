import torch
import numpy as np
import copy
import math 
import cvxpy as cp
import MAT_utils.utils_aux_MAT_attack as MAT_AUX
from learning.MAT_learning.TauGenerator import *

layers_to_split_resnet_generator = [
    'model0.conv1.weight',
    'model0.bn1.weight',
    'model0.bn1.bias',
    'model0.conv2.weight',
    'model0.bn2.weight',
    'model0.bn2.bias',
    "model0.conv3.weight",
    'model0.bn3.weight',
    'model0.bn3.bias',
    'model0.resnet_block1.conv_block.1.weight',
    'model0.resnet_block1.conv_block.2.weight',
    'model0.resnet_block1.conv_block.2.bias',
    'model0.resnet_block1.conv_block.5.weight',
    'model0.resnet_block1.conv_block.6.weight',
    'model0.resnet_block1.conv_block.6.bias',
    'model0.resnet_block2.conv_block.1.weight',
    'model0.resnet_block2.conv_block.2.weight',
    'model0.resnet_block2.conv_block.2.bias',
    'model0.resnet_block2.conv_block.5.weight',
    'model0.resnet_block2.conv_block.6.weight',
    'model0.resnet_block2.conv_block.6.bias',
    'model0.resnet_block3.conv_block.1.weight',
    'model0.resnet_block3.conv_block.2.weight',
    'model0.resnet_block3.conv_block.2.bias',
    'model0.resnet_block3.conv_block.5.weight',
    'model0.resnet_block3.conv_block.6.weight',
    'model0.resnet_block3.conv_block.6.bias',
    'model0.resnet_block4.conv_block.1.weight',
    'model0.resnet_block4.conv_block.2.weight',
    'model0.resnet_block4.conv_block.2.bias',
    'model0.resnet_block4.conv_block.5.weight',
    'model0.resnet_block4.conv_block.6.weight',
    'model0.resnet_block4.conv_block.6.bias',
    'model0.resnet_block5.conv_block.1.weight',
    'model0.resnet_block5.conv_block.2.weight',
    'model0.resnet_block5.conv_block.2.bias',
    'model0.resnet_block5.conv_block.5.weight',
    'model0.resnet_block5.conv_block.6.weight',
    'model0.resnet_block5.conv_block.6.bias',
    'model0.resnet_block6.conv_block.1.weight',
    'model0.resnet_block6.conv_block.2.weight',
    'model0.resnet_block6.conv_block.2.bias',
    'model0.resnet_block6.conv_block.5.weight',
    'model0.resnet_block6.conv_block.6.weight',
    'model0.resnet_block6.conv_block.6.bias',
    'model0.conv4.weight',
    'model0.bn4.weight',
    'model0.bn4.bias',
    'model0.conv5.weight',
    'model0.bn5.weight',
    'model0.bn5.bias',
    'model0.conv6.weight',
    'model0.conv6.bias',
]

layers_to_split_resnet_generator_with_dropout = [
    'model0.conv1.weight',
    'model0.bn1.weight',
    'model0.conv2.weight',
    'model0.bn2.weight',
    'model0.bn2.bias',
    'model0.conv3.weight',
    'model0.bn3.weight',
    'model0.bn3.bias',
    'model0.resnet_block1.conv_block.1.weight',
    'model0.resnet_block1.conv_block.2.weight',
    'model0.resnet_block1.conv_block.2.bias',
    'model0.resnet_block1.conv_block.6.weight',
    'model0.resnet_block1.conv_block.7.weight',
    'model0.resnet_block1.conv_block.7.bias',
    'model0.resnet_block2.conv_block.1.weight',
    'model0.resnet_block2.conv_block.2.weight',
    'model0.resnet_block2.conv_block.2.bias',
    'model0.resnet_block2.conv_block.6.weight',
    'model0.resnet_block2.conv_block.7.weight',
    'model0.resnet_block2.conv_block.7.bias',
    'model0.resnet_block3.conv_block.1.weight',
    'model0.resnet_block3.conv_block.2.weight',
    'model0.resnet_block3.conv_block.2.bias',
    'model0.resnet_block3.conv_block.6.weight',
    'model0.resnet_block3.conv_block.7.weight',
    'model0.resnet_block3.conv_block.7.bias',
    'model0.resnet_block4.conv_block.1.weight',
    'model0.resnet_block4.conv_block.2.weight',
    'model0.resnet_block4.conv_block.2.bias',
    'model0.resnet_block4.conv_block.6.weight',
    'model0.resnet_block4.conv_block.7.weight',
    'model0.resnet_block4.conv_block.7.bias',
    'model0.resnet_block5.conv_block.1.weight',
    'model0.resnet_block5.conv_block.2.weight',
    'model0.resnet_block5.conv_block.2.bias',
    'model0.resnet_block5.conv_block.6.weight',
    'model0.resnet_block5.conv_block.7.weight',
    'model0.resnet_block5.conv_block.7.bias',
    'model0.resnet_block6.conv_block.1.weight',
    'model0.resnet_block6.conv_block.2.weight',
    'model0.resnet_block6.conv_block.2.bias',
    'model0.resnet_block6.conv_block.6.weight',
    'model0.resnet_block6.conv_block.7.weight',
    'model0.resnet_block6.conv_block.7.bias',
    'model0.conv4.weight',
    'model0.bn4.weight',
    'model0.bn4.bias',
    'model0.conv5.weight',
    'model0.bn5.weight',
    'model0.bn5.bias',
    'model0.conv6.weight',
    'model0.conv6.bias'
]

def model_manual_copy(tau, args, gpulist):
    ''''
    Description: 
        - Performs a deep copy of a model and returns the deep copy of a model.
    Arguments: 
        - cross_dimension: (int) Dimension accross which the crossover will be performed
        - num_crossovers: (int) number of crossovers to perform
        - omega: (list) of taus
        - args: args,
        - gpulist: (list) of GPU IDs
        - point_magnet: (bool) If using antipodes method
        - POF: (list) POF points
        - skipper: (int) How many layers are being skipped in every skip
    Returns:
        - new generator model generated based on selected tau. 
    '''
    # Recreate the generator and deep copy state dict
    tau_copy = TauGenerator(args, gpulist)
    tau_copy.model.load_state_dict(tau.model.module.state_dict()) # copy state
    tau_copy.model = torch.nn.DataParallel(tau_copy.model)

    # Optimizer
    optimizer = type(tau.optimizer)(tau_copy.model.parameters(), lr=tau.optimizer.defaults['lr'])
    optimizer.load_state_dict(tau.optimizer.state_dict())
    tau_copy.optimizer = optimizer 
   
    #Characteristics
    tau_copy.model_characteristics = tau.model_characteristics

    # Empirical Losses
    tau_copy.empirical_losses = tau.empirical_losses.copy()
    
    tau_copy.fitness_distance = tau.fitness_distance
    tau_copy.final_fitness_distance = tau.final_fitness_distance
    tau_copy.pof_point_fitness_distance = tau.pof_point_fitness_distance
              
    return tau_copy

def evaluate_model(tau, POF, input_batch_cuda, mean, std, args, mtask_model, target_batch, masks_batch, criteria, \
                    POF_point_index=None,  plain_val = False, del_imgs=True, gradient_step=False):
    ''''
    Description: 
        - Evaluates model by doing forward pass through the attack neural network, the output is then passed through
    the multi-task neural network. Based on the output we obtain losses. Based on the losses we obtain the fitness-distance
    from the pray POF-point.
    Arguments: 
        - tau: (dict) Omega
        - POF: (list) POF points
        - input_batch_cuda: (list) of taus
        - mean: (list) mean of the dataset,
        - std: (list) standard deviation of the dataset
        - args: (args)
        - mtask_model: (object) multi-task neural network model
        - target_batch: 
        - masks_batch:
        - criteria: loss functions
        - POF_point_index: (int)
        - plain_val: (bool) loss with requires_grad = True or False
        - del_imgs: (bool) delete output images or not
        - gradient_step: (bool) is this method used in the gradient step
    Returns:
        - tau with losses and fitness-distance. 
    '''
    tau.optimizer.zero_grad() # zero out the gradients from the previous iteration
    tau = forward_pass_attack_model(tau, input_batch_cuda, mean, std, args, del_imgs=del_imgs)
    tau = compute_losses(tau,mtask_model, target_batch, masks_batch, criteria, plain_value=plain_val, del_imgs=del_imgs)
    if gradient_step:
        tau = single_pof_point_fitness_evaluation(tau,POF, POF_point_index,args)
    return tau

def forward_pass_attack_model(tau, input_batch_cuda, mean, std, args, del_imgs=True):
    ''''
    Description: 
        - Forward pass through the MAT-Attack generator.
    Arguments: 
        - tau: (TauGenerator) Omega
        - input_batch_cuda: (list) of taus
        - mean: (list) mean of the dataset,
        - std: (list) standard deviation of the dataset
        - args: (args)
        - del_imgs: (bool) delete output images or not
    Returns:
        - tau with perturbed images. 
    '''
    tau.model.module.zero_grad()
    tau.output = tau.model.module(input_batch_cuda)

    batch_size = args.batch_size if args.mode == "train" else args.test_batch_size
    tau.output = normalize_and_scale(tau.output, mean_arr=mean, stddev_arr=std, batch_size = batch_size)
    perturbed_images = torch.add(input_batch_cuda, tau.output.cuda(0))

    # do clamping per channel
    for channel in range(3):
        perturbed_images[:,channel,:,:] = perturbed_images[:,channel,:,:].clone().clamp(input_batch_cuda[:,channel,:,:].min(), input_batch_cuda[:,channel,:,:].max())
    
    tau.perturbed_images = perturbed_images   

    # remove tau output
    if del_imgs:
        del tau.output
        torch.cuda.empty_cache()
    return tau

def normalize_and_scale(delta_im, mean_arr, stddev_arr, batch_size):
    ''''
    Description: 
        - Normalizes and scales images.
    Arguments: 
        - delta_im: image
        - mean_arr: (list) mean of the dataset
        - stddev_arr: (list) standard deviation of the dataset
        - batch_size: (int) 
    Returns:
        - normalized and scaled image(s). 
    '''

    delta_im = delta_im + 1 # now 0..2
    delta_im = delta_im * 0.5 # now 0..1

    # normalize image color channels
    for c in range(3):
        delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - mean_arr[c]) / stddev_arr[c]
 
    mag_in = 10.0  #l_inf magnitude of perturbation
    # threshold each channel of each image in deltaIm according to inf norm
    # do on a per image basis as the inf norm of each image could be different
    for i in range(batch_size):
        # do per channel l_inf normalization
        for ci in range(3):
            l_inf_channel = delta_im[i,ci,:,:].detach().abs().max()
            mag_in_scaled_c = mag_in/(255.0*stddev_arr[ci])
            delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())

    return delta_im

def compute_losses(tau,mtask_model, target_batch, masks_batch, criteria, plain_value=False, del_imgs=True):
    ''''
    Description: 
        - Computes losses for each task.
    Arguments: 
        - tau: (TauGenerator) Omega
        - mtask_model: (object) multi-task neural network model
        - target_batch: 
        - masks_batch:
        - criteria: loss functions
        - plain_value: (bool) loss with requires_grad = True or False
        - del_imgs: (bool) delete output images or not
    Returns:
        - Tau with computed losses. 
    '''
    # pass the output of tau through mtask network and get losses
    mtask_output = mtask_model.forward(tau.perturbed_images)
    
    for targeted_task, task_loss_fn in criteria.items():
        # compute specific loss
        # plain value represents if the value is not autograd connected
        if plain_value == False:
            tau.empirical_losses[targeted_task] = task_loss_fn(mtask_output[targeted_task].float(), target_batch[targeted_task], masks_batch[targeted_task])
        elif plain_value == True:
            tau.empirical_losses[targeted_task] = task_loss_fn(mtask_output[targeted_task].float(), target_batch[targeted_task], masks_batch[targeted_task]).item() 
    # get rid of unnecessary pointers to perturbed images 
    if del_imgs:
        del mtask_output
        del tau.perturbed_images
        torch.cuda.empty_cache()

    return tau

def single_pof_point_fitness_evaluation(tau,POF,POF_point_index,args):
    ''''
    Description: 
        - Computes Eulidean distance between single point on POF and a point constructed of losses.
    Arguments: 
        - tau: (TauGenerator) 
        - POF: (list) POF points
        - POF_point_index: POF point used to compute fitness
        - args:
    Returns:
        - Tau with computed pof-point-fitness-distance. 
    '''
    # sort the losses in alphabetical order based on the loss name
    # this is done because we are providin A and b in alphabetical order
    tasks = []
    for letter in args.targeted_tasks_set:
        tasks.append(MAT_AUX.abbrev_to_task(letter)[0])

    point_in_loss_space = []
    # get the point in space
    for task in tasks:
        point_in_loss_space.append(tau.empirical_losses[task])

    sum_of_squares = torch.tensor([0.], requires_grad=True)
    
    # pair_of_points = torch.tensor(POF[POF_point_index]).cuda()
    for index, loss in enumerate(point_in_loss_space):
        if index == 0:
            # doing this to avoid - RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
            sum_of_squares = (torch.tensor([POF[POF_point_index][index]]).cuda() - loss)**2
        else:
            sum_of_squares += (torch.tensor([POF[POF_point_index][index]]).cuda() - loss)**2
    # del pair_of_points
    tau.pof_point_fitness_distance = torch.sqrt(sum_of_squares)
    return tau


def fitness_evaluation_2(omega, A, b, args):
    ''''
    Description: 
        - Computes the distance between a loss points and POF curve. Using cvxpy package for optimization.
    Arguments: 
        - omega: (list) Omega
        - A: matrix representing factors of variables of POF equation
        - b: RHS of the POF equation
        - args:
    Returns:
        - Omega with every member tau evaluated. 
        - probability denominator for the next phase of the algorithm. 
    '''
    # This is a computation for the next step of probability distribution
    # doing it here so we don't have to iterate over all tau again just for 
    # the probability denominator
    probability_denominator = 0
    tasks = []
    for letter in args.targeted_tasks_set:
        tasks.append(MAT_AUX.abbrev_to_task(letter)[0])

    for tau in omega:
        # sort the losses in alphabetical order based on the loss name
        # this is done because we are providin A and b in alphabetical order        
        point_in_loss_space = []
        # get the point in space
        for task in tasks:
            if isinstance(tau.empirical_losses[task],float):
                point_in_loss_space.append(tau.empirical_losses[task])
            else:
                point_in_loss_space.append(tau.empirical_losses[task].item())
                
        # Problem dimensions (m inequalities in n-dimensional space).
        n = len(point_in_loss_space)

        if args.POF_type == "linear":
            # Create variable.
            x_l2 = cp.Variable(shape=n)
            constraints = [A@x_l2 >= b]
        elif args.POF_type == "nonlinear":
            x_l2 = cp.Variable(shape=n, pos=True)
            constraints = [x_l2[0] >= 0.1, cp.inv_prod(x_l2) <= np.prod(A)]

        # Form objective.
        obj = cp.Minimize(cp.norm(x_l2 - point_in_loss_space, 2))

        # Form and solve problem.
        prob = cp.Problem(obj, constraints)
        prob.solve()
        
        tau.final_fitness_distance = obj.value
        # print("For point in loss space: ", point_in_loss_space, "the distance from pareto-optimal front is:", obj.value)
        # computation for the next step
        probability_denominator += math.exp(-1 * args.p_lambda * tau.final_fitness_distance)

    return omega, probability_denominator

def fitness_evaluation_validation(losses, A, b, args):
    '''
    Description: 
        - fitness evaluation in the validation phase
    Arguments:
        - losses: dictionary of losses
        - A: matrix representing factors of variables of POF equation
        - b: RHS of the POF equation
    Returns:
        - fitness based on the losses provided. 
    '''    
    # sort the losses in alphabetical order based on the loss name
    # this is done because we are providin A and b in alphabetical order
    tasks = []
    for letter in args.targeted_tasks_set:
        tasks.append(MAT_AUX.abbrev_to_task(letter)[0])
       
    point_in_loss_space = []
    # get the point in space
    for task in tasks:
        if isinstance(losses[task],float):
            point_in_loss_space.append(losses[task])
        else:
            point_in_loss_space.append(losses[task].item())
            
    # Problem dimensions (m inequalities in n-dimensional space).
    n = len(point_in_loss_space)

    if args.POF_type == "linear":
        # Create variable.
        x_l2 = cp.Variable(shape=n)
        constraints = [A@x_l2 >= b]
    elif args.POF_type == "nonlinear":
        x_l2 = cp.Variable(shape=n, pos=True)
        constraints = [x_l2[0] >= 0.1, cp.inv_prod(x_l2) <= np.prod(A)]

    # Form objective.
    obj = cp.Minimize(cp.norm(x_l2 - point_in_loss_space, 2))

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    return obj.value
       

def multi_mutation(multiple, num_mutations, omega,greedy_approach, args, gpulist, debug):
    '''
    Description: 
        - Mutation phase.
    Arguments: 
        - multiple: (int) number of multiple layers you want to mutate [1-n]
        - num_mutations: (int) number of taus in omega you would like to experience mutation
        - omega: (list) of taus
        - greedy_approach: (bool) fittest models chosen first
        - args: args,
        - gpulist: (list) of GPU IDs
        - debug: (bool) In debug mode or not
    Returns:
        - List of selected taus after they experienced mutation.
    '''
    if debug == True:
        print("\nComputing Mutation...\n")
        
    omega_children = []

    picked_taus = []
    if greedy_approach:
        omega = sorted(omega, key=lambda i: i.final_fitness_distance)
        picked_taus = range(num_mutations)
    else:
        for _ in range(num_mutations):
            random_index = np.random.randint(0, len(omega))
            picked_taus.append(random_index)

    mutation_types = ['random', 'flip']
    for mutation_index, tau in enumerate(picked_taus):
        # create a child
        child_tau = model_manual_copy(omega[tau], args, gpulist)
                    
        # Mutate 1 neuron in each layer
        if args.arch_tau =='resnet' and args.dropout:
            layers_to_mutate = layers_to_split_resnet_generator_with_dropout
        elif args.arch_tau =='resnet':
            layers_to_mutate = layers_to_split_resnet_generator

        for _ in range(multiple):
            # random selection of a layer to mutate
            layer_to_mutate = layers_to_mutate[np.random.randint(0, len(layers_to_mutate))]
            
            mutation_type = mutation_types[mutation_index % (len(mutation_types))]
                                        
            if mutation_type == 'flip':
                child_tau.model.module.state_dict()[layer_to_mutate] = torch.neg(child_tau.model.module.state_dict()[layer_to_mutate])
            else:
                # [-1,1] range
                child_tau.model.module.state_dict()[layer_to_mutate] = 2 * torch.rand(list(child_tau.model.module.state_dict()[layer_to_mutate].size())) - 1

        omega_children.append(child_tau)
    return omega_children

def crossover_skipper2(cross_dimension,num_crossovers, omega,args, gpulist, point_magnet=False, POF=None,skipper=2):
    '''
    Description: 
        - Crossover phase.
    Arguments: 
        - cross_dimension: (int) Dimension accross which the crossover will be performed
        - num_crossovers: (int) number of crossovers to perform
        - omega: (list) of taus
        - args: args,
        - gpulist: (list) of GPU IDs
        - point_magnet: (bool) If using antipodes method
        - POF: (list) POF points
        - skipper: (int) How many layers are being skipped in every skip
    Returns:
        2 new models generated based on selected taus from omega. 
    '''
    if args.arch_tau =='resnet' and args.dropout:
        layers_to_split = layers_to_split_resnet_generator_with_dropout
    elif args.arch_tau =='resnet':
        layers_to_split = layers_to_split_resnet_generator
    else:
        raise  Exception("Crossover is only implemented for Resnet architecture of generator so far.")
        
    crossover_children = []
    
    if point_magnet:
        pof_points_predators = []
        clossest_predators = []

        for i in range(len(POF)):
            pof_points_predators.append([])
            clossest_predators.append([])

        #add the distances of each predator to each point
        for tau in range(len(omega)):
            # measure the distance of tau from all points
            pof_points_predators = update_predator_list(omega[tau],POF, pof_points_predators,args)

        # get the indices of the closest predators for each point
        for point_index,predator_distance_list in enumerate(pof_points_predators):
            clossest_predators[point_index] = sorted(range(len(predator_distance_list)), key=lambda k: predator_distance_list[k])[:num_crossovers]

    # here you can decide how many crossovers you want
    for i in range(num_crossovers):
        if point_magnet:
            index_child_1 = clossest_predators[0][i]
            index_child_2 = clossest_predators[1][i]
        else:
            if not args.greedy_approach:
                index_child_1 = np.random.randint(0, len(omega))
                index_child_2 = np.random.randint(0, len(omega))
            else:
                omega = sorted(omega, key=lambda i: i.final_fitness_distance)
                # greedy approach (taking best models in one direction)
                index_child_1 = i
                index_child_2 = i+1

        # send parents to GPU
        omega[index_child_1].model.module.cuda()
        omega[index_child_2].model.module.cuda()

        # create 2 children models 
        model_child_1 = model_manual_copy(omega[index_child_1],args,gpulist)
        model_child_2 = model_manual_copy(omega[index_child_2],args,gpulist)

        # send parents back to CPU to save GPU memory
        omega[index_child_1].model.module.cpu()
        omega[index_child_2].model.module.cpu()
       
        # print_gpu_obj()
        for layer in layers_to_split:
            indexes = torch.LongTensor([])

            # if it is 1D tensor
            if len(model_child_1.model.module.state_dict()[layer].size()) == 1:
                for i in range(model_child_1.model.module.state_dict()[layer].size()[0]):
                    if i % skipper == 0:
                        indexes = torch.cat((indexes,torch.LongTensor([i])),dim=0)
                    else:
                        indexes = torch.cat((indexes,torch.LongTensor([0])),dim=0)

                temp_model_1 = model_child_1.model.module.state_dict()[layer].detach().clone()
                temp_model_2 = model_child_2.model.module.state_dict()[layer].detach().clone()

                #copying elements of t1 ensor to 'a' in given order of index
                model_child_1.model.module.state_dict()[layer].index_copy_(0,indexes.cuda(),temp_model_2.cuda())
                model_child_2.model.module.state_dict()[layer].index_copy_(0,indexes.cuda(),temp_model_1.cuda())
                
            else:
                for i in range(model_child_1.model.module.state_dict()[layer].size()[cross_dimension]):
                    if i % skipper == 0:
                        indexes = torch.cat((indexes,torch.LongTensor([i])),dim=0)
                    else:
                        indexes = torch.cat((indexes,torch.LongTensor([0])),dim=0)

                temp_model_1 = model_child_1.model.module.state_dict()[layer].detach().clone()
                temp_model_2 = model_child_2.model.module.state_dict()[layer].detach().clone()
                #copying elements of t1 ensor to 'a' in given order of index
                model_child_1.model.module.state_dict()[layer].index_copy_(cross_dimension,indexes.cuda(),temp_model_2.cuda())
                model_child_2.model.module.state_dict()[layer].index_copy_(cross_dimension,indexes.cuda(),temp_model_1.cuda())

                del temp_model_1
                del temp_model_2

            indexes.cpu()
            del indexes
            torch.cuda.empty_cache()
        
        # send modules to cpu
        model_child_1.model.module.cpu()
        model_child_2.model.module.cpu()

        crossover_children.append(model_child_1)
        crossover_children.append(model_child_2)   

    return crossover_children


def update_predator_list(tau,POF, pof_points_predators,args):
    '''
    Description: 
        - Updates the list of the closest predators for each POF point based on the given tau.
    Arguments: 
        - tau: (TauGenerator) 
        - POF: (list) POF points
        - pof_points_predators: (list) 
        - args: args
    Returns:
        2 new models generated based on selected taus from omega. 
    '''
    # sort the losses in alphabetical order based on the loss name
    # this is done because we are providin A and b in alphabetical order
    tasks = []
    for letter in args.targeted_tasks_set:
        tasks.append(MAT_AUX.abbrev_to_task(letter)[0])
        
    point_in_loss_space = []
    # get the point in space
    for task in tasks:
        if isinstance(tau.empirical_losses[task],float):
            point_in_loss_space.append(tau.empirical_losses[task])
        else:
            point_in_loss_space.append(tau.empirical_losses[task].item())

    sum_of_squares = torch.tensor([0.], requires_grad=False)
    for POF_point_index in range(len(POF)):
        for index, loss in enumerate(tau.empirical_losses.values()):
            if index == 0:
                # doing this to avoid - RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
                sum_of_squares = (torch.tensor([POF[POF_point_index][index]]).cuda() - loss)**2
            else:
                sum_of_squares += (torch.tensor([POF[POF_point_index][index]]).cuda() - loss)**2
        pof_points_predators[POF_point_index].append(torch.sqrt(sum_of_squares).item())
    return pof_points_predators

def reproduction(num_selections, omega,  greedy_approach, distances_list,probabilities_list, gpulist, POF,\
                 input_batch_cuda, mean, std, args, mtask_model, target_batch, masks_batch, criteria, experiment):
    '''
    Description: 
        - Performs reproduction phase.
    Arguments: 
        - num_selections: (dict) Omega
        - omega: (list) POF points
        - greedy_approach: (bool) 
        - distances_list: (list) of distances of each tau in omega
        - probabilities_list: (list) of probabilities for each distance in distances_list
        - gpulist: (list) of GPUs
        - POF: (list) POF points
        - input_batch_cuda: (list) of taus
        - mean: (list) mean of the dataset,
        - std: (list) standard deviation of the dataset
        - args: (args)
        - mtask_model: (object) multi-task neural network model
        - target_batch: 
        - masks_batch:
        - criteria: loss functions
        - experiment: comet experiment object
    Returns:
        2 new models generated based on selected taus from omega. 
    '''
    reproduction_taus = []
    for i in range(num_selections):
        if not greedy_approach:
            # get a random distance based on the custom distribution P
            rand_distance = np.random.choice(a = distances_list, p = probabilities_list)

            # choose the tau that has the randomly picked distance determined above
            picked_model = next((tau for tau in omega if tau.final_fitness_distance == rand_distance), None)
        else:
            omega = sorted(omega, key=lambda i: i.final_fitness_distance)
            picked_model = omega[i]

        # send the parent to GPU
        picked_model.model.module.cuda()

        # make a copy of the cihld
        tau_p =  model_manual_copy(picked_model, args, gpulist)

        if args.comet and args.gpu_debug:
            experiment.log_metric("percent_of_GPU_occupied",MAT_AUX.show_gpu(msg="",printer = args.debug_mode))

        # return the parent to CPU
        picked_model.model.module.cpu()

        tau_p = evaluate_model(tau_p, POF, input_batch_cuda, mean, std, args, mtask_model, target_batch, masks_batch, criteria, plain_val=True)

        # send new tau module on CPU since not being evaluated later
        tau_p.model.module.cpu()

        reproduction_taus.append(tau_p)

    return reproduction_taus

def compute_probability_p(omega, probability_denominator, args):
    '''
    Description: 
        - Computes a list of distances for all taus as well as probability for each distance.
    Arguments: 
        - omega: (list) taus
        - probability_denominator: (float) 
        - args: args
    Returns:
        Updated omega, list of distances and list of probabilities. 
    '''
    # lists necessary for distribution sampling
    distances_list = np.array([])
    probabilities_list = np.array([])
    suma = 0
    for i in range(len(omega)):
        omega[i].p_probability = (-1 * args.p_lambda * math.exp(omega[i].final_fitness_distance)) / probability_denominator
         # handling the case when the sum of all probabilities is not 1.0 but for ex. 1.000000000000345345
        suma += omega[i].p_probability
        distances_list = np.append(distances_list, omega[i].final_fitness_distance)
        probabilities_list = np.append(probabilities_list, omega[i].p_probability)
    #  normalize the values so they add up to 1.0. This step necessary becuase of the precision required in np.choice
    probabilities_list = probabilities_list / sum(probabilities_list)  

    if sum(probabilities_list) != 1.0:
        difference = suma - 1.0
        if args.debug_mode:
            print("The difference between sum of probabilities and 1.0 is: ", difference)
    return omega,  distances_list, probabilities_list

def child_gradient_step_point_based(child):
    '''
    Description: 
        - Computes losses PARENT's children models.
            x -> tau -> mtask_nn -> L
            Every child will have only one loss. 
    Arguments: 
        - child: tau
    Returns:
        - Updated child.
    '''
    # Every child will be updating weights based on a single loss.
    # very child will take a different loss function
    child.model.module.train()         
    child.optimizer.zero_grad()
    
    #compute gradients based on one of the fitness losses
    child.pof_point_fitness_distance.backward()

    # NOTE: It will update only child's weights and not mtask model weights since child is defined in the optimizer
    child.optimizer.step()

    # Zero out the gradients for the next step
    child.optimizer.zero_grad()

    # Get rid of perturbed images and the computational graph of the loss to free GPU memory
    del child.perturbed_images
    del child.output
    
    # convert those values to floats since they are containing backward graph that is not needed anymore after the gradient step
    # this will free  necessary gpu memory
    for loss, val in child.empirical_losses.items():
        temp = val.item()
        del  val
        child.empirical_losses[loss]=temp
    
    # child.fitness-distance'][target_point_index] = 
    temp = child.pof_point_fitness_distance.item()
    del  child.pof_point_fitness_distance
    child.pof_point_fitness_distance=temp
    # free the caches gpu memory
    torch.cuda.empty_cache()

    return child

def initiate_children_point_based2(tau, hunting_point_index, gpulist, args): 
    '''
    Description: 
        - Creates children based on huntin point index.
    Arguments: 
        - tau: (TauGenerator)
        - hunting_point_index: (int)
        - gpulist: (list)
        - args
    Returns:
        - Children.
    '''      
    tau_children = []
    for point_index in hunting_point_index:
        # Manual copy of the graph since the model cannot support deepcopy
        child_tau = model_manual_copy(tau, args, gpulist)
        # Make sure gradient tracking is on. This will turn on dropout and batchnorm if used.
        child_tau.model.module.train()
        child_tau.chasing_point_index = point_index
        tau_children.append(child_tau)
    return tau_children

def gradient_step(omega,num_closest_predators, input_batch_cuda, target_batch, masks_batch,criteria,mean,std,mtask_model, \
                     POF,  experiment, args,gpulist):
    ''''
    Description: 
        - Performs gradient step on every model in omega.
    Arguments: 
        - omega: (list) of TauGenerators
        - num_closest_predators: number of closest predators
        - input_batch_cuda: input images
        - target_batch: label images
        - masks_batch: input masks
        - criteria: loss functions
        - mean: (list) mean of the dataset,
        - std: (list) standard deviation of the dataset
        - mtask_model: (object) multi-task neural network model
        - POF: (list) POF points
        - experiment: comet object      
        - args: (args)
        - gpulist: (list) of GPU indices
    Returns:
        - list of TauGenerators on which gradient step is applied.
    '''
    cpu_iterator = 0
    pof_points_predators = []
    clossest_predators = []
    all_tau_children=[]
    for i in range(len(POF)):
        pof_points_predators.append([])
        clossest_predators.append([])

    # measure the distance from each predator tau to each point on POF
    for tau in range(len(omega)):
        # measure the distance of tau from all points
        pof_points_predators = update_predator_list(omega[tau],POF, pof_points_predators,args)

    # get the indices of the closest predators for each point
    for point_index,predator_distance_list in enumerate(pof_points_predators):
        clossest_predators[point_index] = sorted(range(len(predator_distance_list)), \
                                                    key=lambda k: predator_distance_list[k])[:num_closest_predators]
    for i in range(len(omega)):
        # determine if the tau is in the predator list
        hunting_point_index = []
        for indx, predator_list in enumerate(clossest_predators):
            if i in predator_list:
                hunting_point_index.append(indx)
        
        # if tau is not in the predator list, move onto the next tau
        if len(hunting_point_index) == 0:
            continue                     

        tau_children = initiate_children_point_based2(omega[i], hunting_point_index, gpulist, args)  

        if args.comet and args.gpu_debug: experiment.log_metric("percent_of_GPU_occupied", MAT_AUX.show_gpu(msg="",printer = args.debug_mode))
            
        for child_index in range(len(tau_children)):
            tau_children[child_index].model.module.cuda()
  
            tau_children[child_index] = evaluate_model(tau_children[child_index], POF, input_batch_cuda, mean, std, args, mtask_model, target_batch, \
                                                        masks_batch, criteria, del_imgs=False, POF_point_index=tau_children[child_index].chasing_point_index, \
                                                        gradient_step=True)
            tau_children[child_index] = child_gradient_step_point_based(tau_children[child_index])                    
            tau_children[child_index] = evaluate_model(tau_children[child_index], POF, input_batch_cuda, mean, std, args, mtask_model, target_batch, \
                                                                masks_batch, criteria, plain_val=True)      
        
        all_tau_children.extend(tau_children)
        if args.comet and args.gpu_debug: experiment.log_metric("percent_of_GPU_occupied", MAT_AUX.show_gpu(msg="",printer = args.debug_mode))

        # Send children to CPU
        while cpu_iterator < len(tau_children):
            tau_children[cpu_iterator].model.module.cpu()
            cpu_iterator += 1  
    del tau_children
    return all_tau_children