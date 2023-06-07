import torch
from MAT_models.ResNet_MAT import ResnetGenerator, set_init_counter, set_mixed_init, get_init_methods,weights_init

class TauGenerator:
    def __init__(self, args, gpulist, num_blocks=6):
        self.model = ResnetGenerator(3, 3, args.ngf, norm_type=args.norm_type, act_type=args.activ_func, gpu_ids=gpulist, n_blocks=num_blocks, use_dropout=args.dropout)
        self.model_characteristics = {'model':'resnet18'}
        self.optimizer = None
        self.fitness_distance = None
        self.output = None
        self.perturbed_images = None
        self.empirical_losses = {}
        self.pof_point_fitness_distance = None
        self.final_fitness_distance = None
        self.chasing_point_index = None
        self.p_probability = None
        
    def load_weights(self,model_index, args):
        # In case of pretrained model
        if args.pretrained_generator_model_path:
            if args.debug_mode: print("\n\nLoading a pretrained GENERATOR model...\nPretrained model path: ",args.pretrained_generator_model_path)
            self.model.load_state_dict(torch.load(args.pretrained_generator_model_path)) 
        else:
            if model_index % 2 == 1:
                weights_init_mix = True
            else:
                weights_init_mix = False

            if weights_init_mix == True:
                set_mixed_init(True)
                self.model.apply(weights_init)
            else:
                set_mixed_init(False)
                init_method_id = 2 # for xavier_normal_
                set_init_counter(init_method_id)
                self.model.apply(weights_init)

    def write_model_characteristics(self, args,crossover_versions,crossover_dimensions):
        self.model_characteristics['num_reproduction_models'] = args.num_reproduction_models
        self.model_characteristics['crossover_versions'] = crossover_versions
        self.model_characteristics['crossover_dimensions'] = crossover_dimensions
        self.model_characteristics['greedy_approach'] = args.greedy_approach
        self.model_characteristics['optimizer'] = args.optimizer
        self.model_characteristics['lr'] = args.lr
        self.model_characteristics['initialization_method(s)'] = get_init_methods()

    def set_optimizer(self, args):
        if self.model_characteristics['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr,  momentum=0.9 , nesterov=True, dampening=0)
        else:
            raise Exception("Currently, the implementation is only done for SGD optimizer.")