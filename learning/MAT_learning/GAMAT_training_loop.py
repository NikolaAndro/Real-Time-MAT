import torch
import matplotlib.pyplot as plt
import MAT_utils.utils_MAT_attack_minimal as MAT
import MAT_utils.utils_aux_MAT_attack as MAT_AUX

first_iter = True
def GAMAT(input_batch, target_batch, masks_batch, omega, mtask_model, criteria, targeted_tasks, epoch: int, args,\
            mean, std, POF, A, b, gpulist, crossover_versions, crossover_dimensions, good_init_check, POF_to_plot,\
            x_axis_width, y_axis_height, experiment = None):
    '''
    input_batch: input images
    target_batch: labels
    masks_batch: masks for images
    omega: initial N models from stage 1
    optimizers_list: optimizers for each of the N models
    mtask_model: Multi-Task neural network model
    criteria: loss functions
    targeted_tasks: tasks being attacked
    epoch:  epoch
    args: args
    experiment: comet experiment object
    '''
    global first_iter

    # send data to cuda if available
    if torch.cuda.is_available():
        #check if model is already on GPU
        while True:
            item = next(mtask_model.parameters()).is_cuda
            break
        if item == False:
            mtask_model = mtask_model.cuda()

        if input_batch.is_cuda == False:
            input_batch = input_batch.cuda()
            input_batch_cuda = torch.autograd.Variable(input_batch)
        
        for keys, tar in target_batch.items():
            if tar.is_cuda == False:
                target_batch[keys] = tar.cuda()
        for keys, m in masks_batch.items():
            if m.is_cuda == False:
                masks_batch[keys] = m.cuda()

    for tau in range(len(omega)):  # forward pass for all the models in omega and plot their losses  
        # if 'fitness-distance' not in tau.keys():
        omega[tau].model.module.cuda()
        if args.comet and args.gpu_debug: experiment.log_metric("percent_of_GPU_occupied", MAT_AUX.show_gpu(msg="",printer = args.debug_mode))  
        omega[tau] = MAT.evaluate_model(omega[tau], POF, input_batch_cuda, mean, std, args, mtask_model, target_batch,\
                                            masks_batch, criteria, plain_val=True)
        omega[tau].model.module.cpu()

    omega, probability_denominator = MAT.fitness_evaluation_2(omega, A,b, args)

    if good_init_check == False and args.save_plots:
        fig = plt.figure(0)
        MAT_AUX.plot_POF(POF_to_plot, POF,x_axis_width)
        MAT_AUX.plot_losses('initial models' if not args.single_color else 'MAT', omega, targeted_tasks, "k",\
                args.experiment_backup_folder +"iterations/init_models_losses.pdf", 1.4, args)
        response = None
        if args.debug_mode:
            #check the losses and if they are diverse enough, choose to continue the process
            while response not in ('y','Y','N','n'):
                response = input("Are you satisfied with initial losses? (y/n)") 
            if response == 'n':
                exit()
        else:
            good_init_check = True
    
    for test_one_imege_iter in range(args.num_loop_iters): # used in single image tests
        # If doing mini tests, every iteration on a single batch will be saved as another plot.
        figure_index = test_one_imege_iter if test_one_imege_iter > 0 else epoch
            
        all_tau_children=[]
            
        #**************************************STAGE 2.1 - SELECTIVE GRADIENT-BASED UPDATE **************************************
        all_tau_children = MAT.gradient_step(omega, args.num_grad_predators,input_batch_cuda, target_batch, masks_batch,criteria,mean,std,mtask_model,\
                                            POF, experiment, args,gpulist)

        all_tau_children, probability_denominator = MAT.fitness_evaluation_2(all_tau_children, A,b, args)
        omega.extend(all_tau_children) # Add children to parents Omega

        if args.save_plots:#and first_iter:
            # print("\nSUCCESS - children based on a single gradient generated in ",grad_total_time," seconds\n")
            MAT_AUX.plot_losses('gradient models' if not args.single_color else 'MAT', all_tau_children, targeted_tasks, 
            "c" if not args.single_color else 'k', args.experiment_backup_folder +"iterations/"+str(figure_index)+"_grad_models_losses.pdf", y_axis_height, args)
           
        #free memory
        # del tau_children
        del all_tau_children
                    
        # *********************************** STAGE 2.2 - Reproduction from P-distribution *******************************
        omega, distances_list, probabilities_list = MAT.compute_probability_p(omega,probability_denominator, args)
        
        if args.comet and args.gpu_debug: experiment.log_metric("percent_of_GPU_occupied", MAT_AUX.show_gpu(msg="",printer = args.debug_mode))
        
        reproduction_children = MAT.reproduction(args.num_reproduction_models, omega, args.greedy_approach, distances_list,probabilities_list, gpulist, \
            POF, input_batch_cuda, mean, std, args, mtask_model, target_batch, masks_batch, criteria, experiment)   
        
        reproduction_children, _ = MAT.fitness_evaluation_2(reproduction_children, A,b, args)
        omega.extend(reproduction_children)

        if args.save_plots:
            MAT_AUX.plot_losses('reproduction models' if not args.single_color else 'MAT', reproduction_children, targeted_tasks, \
                "r" if not args.single_color else 'k', args.experiment_backup_folder +"iterations/"+str(figure_index)+"_reproduction_models_losses.pdf", y_axis_height, args)

        del reproduction_children
        # *********************************** STAGE 2.3 - Crossover *******************************        
        num_crossovers = len(omega)//2
        if 1 in crossover_versions:
            if args.debug_mode: print("Crossover crossover_skipper2 with ANTIPODES executing...")
            cross_children_brutal_2 = MAT.crossover_skipper2(crossover_dimensions[crossover_versions.index(1)],num_crossovers, omega,args,\
                                                            gpulist, point_magnet=True, POF=POF)
            for tau in cross_children_brutal_2:
                tau.model.module.cuda()
                tau = MAT.evaluate_model(tau, POF, input_batch_cuda, mean, std, args, mtask_model, target_batch, masks_batch, criteria,\
                                                plain_val=True)
                tau.model.module.cpu()
            cross_children_brutal_2, _ = MAT.fitness_evaluation_2(cross_children_brutal_2, A,b,args)
            omega.extend(cross_children_brutal_2)
            
            if args.save_plots:
                MAT_AUX.plot_losses('crossover models' if not args.single_color else 'MAT', cross_children_brutal_2, targeted_tasks, "springgreen",\
                    args.experiment_backup_folder +"iterations/"+str(figure_index)+"_cross_skipper_point_magnet_models_losses.pdf", y_axis_height, args)
                MAT_AUX.plot_losses('MAT', cross_children_brutal_2, targeted_tasks, "k", \
                    args.experiment_backup_folder +"iterations/cross/"+str(figure_index)+"_cross_skipper_point_magnet_models_losses.pdf", y_axis_height, args)
            del cross_children_brutal_2 #free the memory
        if 2 in crossover_versions:
            if args.debug_mode: print("Crossover crossover_skipper2 executing...")
            crossover_children = MAT.crossover_skipper2(crossover_dimensions[crossover_versions.index(2)],num_crossovers, omega, args, gpulist) 
            for tau in crossover_children:
                tau.model.module.cuda()
                tau = MAT.evaluate_model(tau, POF, input_batch_cuda, mean, std, args, mtask_model, target_batch, masks_batch, criteria,\
                                            plain_val=True)
                tau.model.module.cpu()
            crossover_children, _ = MAT.fitness_evaluation_2(crossover_children, A,b, args)
            omega.extend(crossover_children)
            if args.save_plots:
                MAT_AUX.plot_losses('crossover_models' if not args.single_color else 'MAT', crossover_children, targeted_tasks, "blueviolet" if not args.single_color else 'k',\
                    args.experiment_backup_folder +"iterations/"+str(figure_index)+"_cross_skipper_models_losses.pdf", y_axis_height, args)
                MAT_AUX.plot_losses('MAT', crossover_children, targeted_tasks, "k", \
                    args.experiment_backup_folder +"iterations/cross/"+str(figure_index)+"_cross_skipper_models_losses.pdf", y_axis_height, args)
            del crossover_children
        
        torch.cuda.empty_cache() # empty the cache after deleting

        # *********************************** STAGE 2.4 - Mutation ***************************************
        if args.include_mutation:  
            mutation_children = MAT.multi_mutation(args.mutation_multiple,args.num_mutations, omega, args.greedy_approach, \
                                                    args, gpulist, debug=args.debug_mode)
            for tau in mutation_children:
                tau.model.module.cuda()
                tau = MAT.evaluate_model(tau, POF, input_batch_cuda, mean, std, args, mtask_model, \
                                                target_batch, masks_batch, criteria, plain_val=True)
                tau.model.module.cpu()

            mutation_children, _ = MAT.fitness_evaluation_2(mutation_children, A,b,args)   
            omega.extend(mutation_children)
                            
            if args.save_plots:
                MAT_AUX.plot_losses('mutation_models' if not args.single_color else 'MAT', mutation_children, targeted_tasks, \
                    "darkorange" if not args.single_color else 'k', args.experiment_backup_folder +"iterations/"+str(figure_index)+"_mutant_models_losses.pdf", y_axis_height, args)
                
            # free memory
            del mutation_children
            torch.cuda.empty_cache()
       
        # *********************************** STAGE 3 - Rank-Based Elitism  *******************************
        omega = sorted(omega, key=lambda i: i.final_fitness_distance)      

        # Remove all models not in top 10
        for i in reversed(range(args.num_carryovers, len(omega))):
            del omega[i].model.module
            del omega[i].optimizer
            del omega[i]
        torch.cuda.empty_cache()  
            
        if args.save_plots:
            plt.cla() # clean the plot for this iteration for the next one
            # MAT_AUX.plot_POF(POF_to_plot, POF, x_axis_width)
            # MAT_AUX.plot_losses(omega, targeted_tasks, "k", \
            #     args.experiment_backup_folder +"iterations/"+str(figure_index)+"_end_iter_models_losses.pdf", y_axis_height, args)       
    return omega