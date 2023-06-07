# Code borrowed from : https://github.com/OmidPoursaeed/Generative_Adversarial_Perturbations
from typing import OrderedDict
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import numpy as np
import math
import random

init_counter = 0

mixed_initialization = True
init_methods = []

def set_init_counter(val):
    global init_counter
    init_counter = val

def set_mixed_init(mixed_or_not):
    global mixed_initialization
    mixed_initialization = mixed_or_not 

def get_init_methods():
    global init_methods
    return init_methods

def reset_init_methods():
    global init_methods
    init_methods = []

def weights_init(m, act_type='relu'):
    ''' 0: normal_
        1: xavier_uniform_
        2: xavier_normal_
        3: uniform_
        4: constant_
        5: ones_
        6: zeros_
        7: dirac_
        8: kaiming_uniform_
        9: kaiming_normal_
        10: trunc_normal_
        11: orthogonal_'''

    classname = m.__class__.__name__

    initialization_dict = {
        0: 'normal_',
        1: 'xavier_uniform_',
        2: 'xavier_normal_',
        3: 'uniform_',
        4: 'constant_',
        5: 'ones_',
        6: 'zeros_',
        7: 'dirac_',
        8: 'kaiming_uniform_',
        9: 'kaiming_normal_',
        10: 'trunc_normal_',
        11: 'orthogonal_',
    }

    global mixed_initialization
    global init_methods
    
    if mixed_initialization == True:
        # selecting a random initialization
        initialization_counter = np.random.randint(0, 12)
        # print("\n\nWeights initialization is chosen randomly and it is:",initialization_dict[initialization_counter],"\n\n")
    else:
        global init_counter
        initialization_counter = init_counter
        # print("\n\nWeights initialization is chosen randomly and it is:",initialization_dict[initialization_counter],"\n\n")

    init_methods.append(initialization_dict[initialization_counter])

    stds = torch.arange(0,10,0.1)
    means = torch.arange(0,10,0.01)

    if initialization_counter == 0:
        # print("Initializing weights using normal distribution.")
        if classname.find('Conv') != -1:
            if act_type == 'selu':
                n = float(m.in_channels * m.kernel_size[0] * m.kernel_size[1])
                m.weight.data.normal_(0.0, 1.0 / math.sqrt(n))
            else:
                # m.weight.data.normal_(0.0, 0.02) 
                rand_index_std = random.randint(0,len(stds)-1) 
                rand_index_mean = random.randint(0,len(means)-1) 
                m.weight.data.normal_(stds[rand_index_std], means[rand_index_mean])   

            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            # m.weight.data.normal_(1.0, 0.02)
            rand_index_std = random.randint(0,len(stds)-1) 
            rand_index_mean = random.randint(0,len(means)-1) 
            m.weight.data.normal_(std=stds[rand_index_std], mean=means[rand_index_mean])
            m.bias.data.fill_(0)
        
    if initialization_counter == 1:
        # print("Initializing weights using xavier uniform distribution.")
        if classname.find('Conv') != -1:
            init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            # m.weight.data.normal_(1.0, 0.02)
            rand_index_std = random.randint(0,len(stds)-1) 
            rand_index_mean = random.randint(0,len(means)-1) 
            m.weight.data.normal_(std=stds[rand_index_std], mean=means[rand_index_mean])
            m.bias.data.fill_(0)

    elif initialization_counter == 2:
        if classname.find('Conv') != -1:
            init.xavier_normal_(m.weight)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            # m.weight.data.normal_(1.0, 0.02)
            rand_index_std = random.randint(0,len(stds)-1) 
            rand_index_mean = random.randint(0,len(means)-1) 
            m.weight.data.normal_(std=stds[rand_index_std], mean=means[rand_index_mean])
            m.bias.data.fill_(0)

    elif initialization_counter == 3:
        if classname.find('Conv') != -1:
            init.uniform_(m.weight)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            init.uniform_(m.weight)
         
    elif initialization_counter == 4:
        if classname.find('Conv') != -1:
            # init.constant_(m.weight, 0.3)
            random_constant = random.randint(0,len(stds)-1) 
            init.constant_(m.weight, stds[random_constant])
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            # init.constant_(m.weight, 0.3)
            random_constant = random.randint(0,len(stds)-1) 
            init.constant_(m.weight, stds[random_constant])

    elif initialization_counter == 5:
        if classname.find('Conv') != -1:
            init.ones_(m.weight)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            init.ones_(m.weight)
         

    elif initialization_counter == 6:
        if classname.find('Conv') != -1:
            init.zeros_(m.weight)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            init.zeros_(m.weight)

    elif initialization_counter == 7:
        if classname.find('Conv') != -1:
            init.dirac_(m.weight)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            # m.weight.data.normal_(1.0, 0.02)
            rand_index_std = random.randint(0,len(stds)-1) 
            rand_index_mean = random.randint(0,len(means)-1) 
            m.weight.data.normal_(std=stds[rand_index_std], mean=means[rand_index_mean])
            m.bias.data.fill_(0)
         
    elif initialization_counter == 8:
        if classname.find('Conv') != -1:
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            # m.weight.data.normal_(1.0, 0.02)
            rand_index_std = random.randint(0,len(stds)-1) 
            rand_index_mean = random.randint(0,len(means)-1) 
            m.weight.data.normal_(std=stds[rand_index_std], mean=means[rand_index_mean])
            m.bias.data.fill_(0)
         
    elif initialization_counter == 9:
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            # m.weight.data.normal_(1.0, 0.02)
            rand_index_std = random.randint(0,len(stds)-1) 
            rand_index_mean = random.randint(0,len(means)-1) 
            m.weight.data.normal_(std=stds[rand_index_std], mean=means[rand_index_mean])
            m.bias.data.fill_(0)
          
    elif initialization_counter == 10:
        if classname.find('Conv') != -1:
            init.trunc_normal_(m.weight)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            init.trunc_normal_(m.weight)
           
    elif initialization_counter == 11:
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            # m.weight.data.normal_(1.0, 0.02)
            rand_index_std = random.randint(0,len(stds)-1) 
            rand_index_mean = random.randint(0,len(means)-1) 
            m.weight.data.normal_(std=stds[rand_index_std], mean=means[rand_index_mean])
            m.bias.data.fill_(0)
         
def get_scheduler(optimizer, lr_policy, lr_decay_iters = 1):
    if lr_policy == 'LambdaLR':
        def lambda_rule(epoch):
            lr_l = ((0.5 ** int(epoch >= 2)) *
                    (0.5 ** int(epoch >= 5)) *
                    (0.5 ** int(epoch >= 8)))
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_iters, gamma=0.1
        )
    elif lr_policy == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5
        )
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type = 'batch', act_type='selu', use_dropout=False, n_blocks=6, padding_type='reflect', gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpulist = gpu_ids
        self.num_gpus = len(self.gpulist)
        self.all_layers = nn.ModuleList([])

        # self.__getattr__()
        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        model0 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=use_bias),
                  norm_layer(ngf),
                  self.act]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       self.act]

        if self.num_gpus == 1:
            mult = 2**n_downsampling
            for i in range(n_blocks):
                model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        elif self.num_gpus == 2:
            model1 = []
            mult = 2**n_downsampling
            mid = int(n_blocks / 2)
            for i in range(mid):
                model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            for i in range(n_blocks - mid):
                model1 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        elif self.num_gpus == 3:
            model1 = []
            model2 = []
            mult = 2**n_downsampling
            mid1 = int(n_blocks / 5)
            mid2 = mid1 + int((n_blocks - mid1) / 4.0 * 3)
            # mid = int(n_blocks / 2)
            for i in range(mid1):
                model0 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            for i in range(mid1, mid2):
                model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            for i in range(mid2, n_blocks):
                model1 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        if self.num_gpus >= 2:
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                model1 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        self.act]
            model1 += [nn.ReflectionPad2d(3)]
            model1 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model1 += [nn.Tanh()]
        else:
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                model0 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=use_bias),
                        norm_layer(int(ngf * mult / 2)),
                        self.act]
            model0 += [nn.ReflectionPad2d(3)]
            model0 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model0 += [nn.Tanh()] 

        self.all_layers.extend(model0)

        if n_blocks == 7: 
            dict= OrderedDict([
                ('refpad1'      , self.all_layers[0]),
                ('conv1'        , self.all_layers[1]),
                ('bn1'          , self.all_layers[2]),
                ('relu1'        , self.all_layers[3]),
                ('conv2'        , self.all_layers[4]),
                ('bn2'          , self.all_layers[5]),
                ('relu2'        , self.all_layers[6]),
                ('conv3'        , self.all_layers[7]),
                ('bn3'          , self.all_layers[8]),
                ('relu3'        , self.all_layers[9]),
                ('resnet_block1',self.all_layers[10]),
                ('resnet_block2',self.all_layers[11]),
                ('resnet_block3',self.all_layers[12]),
                ('resnet_block4',self.all_layers[13]),
                ('resnet_block5',self.all_layers[14]),
                ('resnet_block6',self.all_layers[15]),
                ('resnet_block7',self.all_layers[16]),
                ('conv4'       , self.all_layers[17]),
                ('bn4'         , self.all_layers[18]),
                ('relu4'       , self.all_layers[19]),
                ('conv5'       , self.all_layers[20]),
                ('bn5'         , self.all_layers[21]),
                ('relu5'       , self.all_layers[22]),
                ('refpad2'     , self.all_layers[23]),
                ('conv6'       , self.all_layers[24]),
                ('tanh1'       , self.all_layers[25])
            ])
        elif n_blocks == 6:   
            dict= OrderedDict([
                ('refpad1'      , self.all_layers[0]),
                ('conv1'        , self.all_layers[1]),
                ('bn1'          , self.all_layers[2]),
                ('relu1'        , self.all_layers[3]),
                ('conv2'        , self.all_layers[4]),
                ('bn2'          , self.all_layers[5]),
                ('relu2'        , self.all_layers[6]),
                ('conv3'        , self.all_layers[7]),
                ('bn3'          , self.all_layers[8]),
                ('relu3'        , self.all_layers[9]),
                ('resnet_block1',self.all_layers[10]),
                ('resnet_block2',self.all_layers[11]),
                ('resnet_block3',self.all_layers[12]),
                ('resnet_block4',self.all_layers[13]),
                ('resnet_block5',self.all_layers[14]),
                ('resnet_block6',self.all_layers[15]),
                ('conv4'       , self.all_layers[16]),
                ('bn4'         , self.all_layers[17]),
                ('relu4'       , self.all_layers[18]),
                ('conv5'       , self.all_layers[19]),
                ('bn5'         , self.all_layers[20]),
                ('relu5'       , self.all_layers[21]),
                ('refpad2'     , self.all_layers[22]),
                ('conv6'       , self.all_layers[23]),
                ('tanh1'       , self.all_layers[24])
            ])
        elif n_blocks == 5:
            dict= OrderedDict([
                ('refpad1'      , self.all_layers[0]),
                ('conv1'        , self.all_layers[1]),
                ('bn1'          , self.all_layers[2]),
                ('relu1'        , self.all_layers[3]),
                ('conv2'        , self.all_layers[4]),
                ('bn2'          , self.all_layers[5]),
                ('relu2'        , self.all_layers[6]),
                ('conv3'        , self.all_layers[7]),
                ('bn3'          , self.all_layers[8]),
                ('relu3'        , self.all_layers[9]),
                ('resnet_block1',self.all_layers[10]),
                ('resnet_block2',self.all_layers[11]),
                ('resnet_block3',self.all_layers[12]),
                ('resnet_block4',self.all_layers[13]),
                ('resnet_block5',self.all_layers[14]),
                ('conv4'       , self.all_layers[15]),
                ('bn4'         , self.all_layers[16]),
                ('relu4'       , self.all_layers[17]),
                ('conv5'       , self.all_layers[18]),
                ('bn5'         , self.all_layers[19]),
                ('relu5'       , self.all_layers[20]),
                ('refpad2'     , self.all_layers[21]),
                ('conv6'       , self.all_layers[22]),
                ('tanh1'       , self.all_layers[23])
            ])

        self.model0 = nn.Sequential(dict)
        self.model0.cuda(self.gpulist[0])
        if self.num_gpus == 2:
            self.model1 = nn.Sequential(*model1)
            self.model1.cuda(self.gpulist[1])
        if self.num_gpus == 3:
            self.model2 = nn.Sequential(*model2)
            self.model2.cuda(self.gpulist[2])

    def forward(self, input):
        input = input.cuda(self.gpulist[0])
        input = self.model0(input)
        if self.num_gpus == 3:
            input = input.cuda(self.gpulist[2])
            input = self.model2(input)
        if self.num_gpus == 2:
            input = input.cuda(self.gpulist[1])
            input = self.model1(input)
        return input

    def selfdestruction(self):
        for layer in self.model0:
            del layer
        del self.model0
        return

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# mean_arr = [-0.9235, -0.9235, -0.9235]#[0.485, 0.456, 0.406]
# stddev_arr = [0.2189, 0.2189, 0.2189]#[0.229, 0.224, 0.225]
mag_in = 10.0

def normalize_and_scale(delta_im, mean_arr, stddev_arr, bs, mode='train' ):
    # if opt.foolmodel == 'incv3':
    #     delta_im = nn.ConstantPad2d((0,-1,-1,0),0)(delta_im) # crop slightly to match inception

    delta_im = delta_im + 1 # now 0..2
    delta_im = delta_im * 0.5 # now 0..1

    # normalize image color channels
    for c in range(3):
        delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - mean_arr[c]) / stddev_arr[c]

    # threshold each channel of each image in deltaIm according to inf norm
    # do on a per image basis as the inf norm of each image could be different
    # bs = 1 # opt.batchSize if (mode == 'train') else opt.testBatchSize
    for i in range(bs):
        # do per channel l_inf normalization
        for ci in range(3):
            l_inf_channel = delta_im[i,ci,:,:].detach().abs().max()
            mag_in_scaled_c = mag_in/(255.0*stddev_arr[ci])
            gpu_id = 0
            delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())

    return delta_im

def dataset_mean_and_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0,0,0

    for input_batch, labels, masks in loader:
        channels_sum += torch.mean(input_batch, dim=[0,2,3]) 
        channels_squared_sum += torch.mean(input_batch**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    return mean, std
    