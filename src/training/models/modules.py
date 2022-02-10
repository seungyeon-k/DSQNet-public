import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(s_act):
    if s_act == 'relu':
        return nn.ReLU()
    elif s_act == 'sigmoid':
        return nn.Sigmoid()
    elif s_act == 'softplus':
        return nn.Softplus()
    elif s_act == 'linear':
        return None
    elif s_act == 'tanh':
        return nn.Tanh()
    elif s_act == 'leakyrelu':
        return nn.LeakyReLU(0.2)
    elif s_act == 'softmax':
        return nn.Softmax(dim=1)
    else:
        raise ValueError(f'Unexpected activation: {s_act}')

class MLP(nn.Module):
    def __init__(self, **args):
        super(MLP, self).__init__()
        self.args = args
        self.l_hidden = args['l_hidden']
        self.output_dim = args['output_dim']
        self.input_dim = args['input_dim']
        self.activation = args['activation']
        self.out_activation = args['out_activation']
        self.leakyrelu_slope = args['leakyrelu_slope']
        if 'use_batch_norm' in args:
            self.use_batch_norm = args['use_batch_norm']
        else:
            self.use_batch_norm = False

        l_neurons = self.l_hidden + [self.output_dim]
        # activation = ['relu']*len(l_neurons)
        
        l_layer = []
        prev_dim = self.input_dim
        i = 0
        for n_hidden in l_neurons:
            i += 1
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            if self.use_batch_norm:
                l_layer.append(nn.BatchNorm1d(n_hidden))
            if i == len(l_neurons):
                act_fn = get_activation(self.out_activation)
                if self.out_activation == 'leakyrelu':
                    act_fn.negative_slope = self.leakyrelu_slope 
            else:
                act_fn = get_activation(self.activation)
                if self.activation == 'leakyrelu':
                    act_fn.negative_slope = self.leakyrelu_slope 
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden 

            self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        x = self.net(x)
        return x


class Pointwise_MLP(nn.Module):
    def __init__(self, **args):
        super(Pointwise_MLP, self).__init__()
        self.args = args
        self.l_hidden = args['l_hidden']
        self.output_dim = args['output_dim']
        self.input_dim = args['input_dim']
        self.activation = args['activation']
        self.out_activation = args['out_activation']
        self.leakyrelu_slope = args['leakyrelu_slope']
        if 'use_batch_norm' in args:
            self.use_batch_norm = args['use_batch_norm']
        else:
            self.use_batch_norm = False

        l_neurons = self.l_hidden + [self.output_dim]
        # activation = ['relu']*len(l_neurons)
        
        l_layer = []
        prev_dim = self.input_dim
        i = 0
        for n_hidden in l_neurons:
            i += 1
            l_layer.append(nn.Conv1d(
                prev_dim, n_hidden, kernel_size=1, bias=False))
            if self.use_batch_norm:
                l_layer.append(nn.BatchNorm1d(n_hidden))
            if i == len(l_neurons):
                act_fn = get_activation(self.out_activation)
                if self.out_activation == 'leakyrelu':
                    act_fn.negative_slope = self.leakyrelu_slope 
            else:
                act_fn = get_activation(self.activation)
                if self.activation == 'leakyrelu':
                    act_fn.negative_slope = self.leakyrelu_slope 
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden 

            self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        x = self.net(x)
        return x

class Estimator_pos_con(nn.Module):
    def __init__(self, **args):
        super(Estimator_pos_con, self).__init__()

    def forward(self, x):
        x_pos = x[:, :3, :]
        sigmoid = get_activation('sigmoid')
        x_confidence = sigmoid(x[:, 3:, :])
        x = torch.cat([x_pos, x_confidence], dim=1)
        return x

class Estimator_pos_nori_con(nn.Module):
    def __init__(self, **args):
        super(Estimator_pos_nori_con, self).__init__()

    def forward(self, x):
        x_pos = x[:, :3, :]
        x_ori = x[:, 3:6, :]
        x_ori = F.normalize(x_ori, p=2, dim=1)
        sigmoid = get_activation('sigmoid')
        x_confidence = sigmoid(x[:, 6:, :])
        x = torch.cat([x_pos, x_ori, x_confidence], dim=1)
        return x

class Estimator_total(nn.Module):
    def __init__(self, **args):
        super(Estimator_total, self).__init__()
        self.len_physical_params = args['len_physical_params']

    def forward(self, x):
        x_pos = x[:, :3, :]
        x_ori = F.normalize(x[:, 3:6, :], p=2, dim=1)
        relu = get_activation('relu')
        x_param = relu(x[:, 6:6+self.len_physical_params, :]) + 1e-5
        sigmoid = get_activation('sigmoid')
        x_confidence = sigmoid(x[:, 6+self.len_physical_params:, :])
        x = torch.cat([x_pos, x_ori, x_param, x_confidence], dim=1)
        return x

class Estimator_sq(nn.Module):
    def __init__(self, **args):
        super(Estimator_sq, self).__init__()

    def forward(self, x):
        x_pos = x[:, :3]
        x_ori = F.normalize(x[:, 3:7], p=2, dim=1)
        relu = get_activation('relu')
        sigmoid = get_activation('sigmoid')
        x_param_size = 0.5 * sigmoid(x[:, 7:10]) + 0.03
        x_param_shape = 1.5 * sigmoid(x[:, 10:12]) + 0.2
        # x_param_shape = relu(x[:, 10:12]) - relu(x[:, 10:12] - 1.9) + 0.1
        x = torch.cat([x_pos, x_ori, x_param_size, x_param_shape], dim=1)
        return x

# class Estimator_sq_local_global(nn.Module):
#     def __init__(self, **args):
#         super(Estimator_sq_local_global, self).__init__()

#     def forward(self, x):
#         x = torch.mean(x, dim=2)
#         x_pos = x[:, :3]
#         x_ori = F.normalize(x[:, 3:7], p=2, dim=1)
#         relu = get_activation('relu')
#         sigmoid = get_activation('sigmoid')
#         x_param_size = relu(x[:, 7:10]) + 1e-1
#         x_param_shape = 1.9 * sigmoid(x[:, 10:12]) + 0.3
#         # x_param_shape = relu(x[:, 10:12]) - relu(x[:, 10:12] - 1.9) + 0.1
#         x = torch.cat([x_pos, x_ori, x_param_size, x_param_shape], dim=1)
#         return x

class Estimator_esq(nn.Module):
    def __init__(self, **args):
        super(Estimator_esq, self).__init__()

    def forward(self, x):
        x_pos = x[:, :3]
        x_ori = F.normalize(x[:, 3:7], p=2, dim=1)
        relu = get_activation('relu')
        sigmoid = get_activation('sigmoid')
        x_param_size = 1.0 * sigmoid(x[:, 7:10]) + 0.03
        x_param_shape = 1.8 * sigmoid(x[:, 10:12]) + 0.2
        x_param_c1 = 1.4 * sigmoid(x[:, 12:13]) - 0.9
        x_param_c2 = 0.5 * sigmoid(x[:, 13:14]) + 0.5
        x = torch.cat([x_pos, x_ori, x_param_size, x_param_shape, x_param_c1, x_param_c2], dim=1)
        return x

class Estimator_st(nn.Module):
    def __init__(self, **args):
        super(Estimator_st, self).__init__()

    def forward(self, x):
        x_pos = x[:, :3]
        x_ori = F.normalize(x[:, 3:7], p=2, dim=1)
        relu = get_activation('relu')
        leakyrelu = get_activation('leakyrelu')
        sigmoid = get_activation('sigmoid')
        # x_param_size_12 = 0.1 * relu(x[:, 7:9]) + 0.0003
        # x_param_size_12 = relu(x[:, 7:9]) + 0.0003
        # x_param_size_3 = 1.0 * sigmoid(x[:, 9:10]) + 0.003

        x_param_size = 0.7 * sigmoid(x[:, 7:10]) + 0.003
        # x_param_shape = 1.5 * sigmoid(x[:, 10:12]) + 0.2

        # x_param_size_radius = 5.0 + leakyrelu(x[:, 10:11])
        x_param_size_radius = leakyrelu(x[:, 10:11])
        # x_param_size_radius = 2.0 + leakyrelu(x[:, 10:11])
        # x_param_shape = 1.5 * sigmoid(x[:, 11:13]) + 0.2
        x_param_shape = 1.8 * sigmoid(x[:, 11:13]) + 0.2
        # x = torch.cat([x_pos, x_ori, x_param_size_12, x_param_size_3, x_param_size_radius, x_param_shape], dim=1)
        x = torch.cat([x_pos, x_ori, x_param_size, x_param_size_radius, x_param_shape], dim=1)
        return x

class Estimator_est(nn.Module):
    def __init__(self, **args):
        super(Estimator_est, self).__init__()

    def forward(self, x):
        x_pos = x[:, :3]
        x_ori = F.normalize(x[:, 3:7], p=2, dim=1)
        relu = get_activation('relu')
        leakyrelu = get_activation('leakyrelu')
        sigmoid = get_activation('sigmoid')
        x_param_size = 0.1 * sigmoid(x[:, 7:10]) + 0.03
        x_param_size_radius = 3.0 + leakyrelu(x[:, 10:11])
        # x_param_size_radius = 2.0 + leakyrelu(x[:, 10:11])
        # x_param_shape = 1.5 * sigmoid(x[:, 11:13]) + 0.2
        x_param_shape = 1.8 * sigmoid(x[:, 11:13]) + 0.2
        x_param_c2 = sigmoid(x[:, 13:14]) - 0.5
        x = torch.cat([x_pos, x_ori, x_param_size, x_param_size_radius, x_param_shape, x_param_c2], dim=1)
        return x

class Individual_MLP(nn.Module):
    def __init__(self, **args):
        super(Individual_MLP, self).__init__()
        self.args = args
        self.input_dim = args['input_dim']
        self.module_collection = args['module_collection']
        for key in self.module_collection.keys():
            self.module_collection[key]['input_dim'] = self.input_dim

        self.net_position = MLP(**self.module_collection['position'])
        self.net_orientation = MLP(**self.module_collection['orientation'])
        self.net_size = MLP(**self.module_collection['size'])
        self.net_shape = MLP(**self.module_collection['shape'])
        # self.net_slice = MLP(**self.module_collection['slice'])

        if 'taper' in self.module_collection.keys():
            self.net_taper = MLP(**self.module_collection['taper'])

        if 'bending' in self.module_collection.keys():
            self.net_bend = MLP(**self.module_collection['bending'])
            self.net_bend_angle = MLP(**self.module_collection['bending_angle'])

    def forward(self, x):

        # activations
        relu = get_activation('relu')
        leakyrelu = get_activation('leakyrelu')
        sigmoid = get_activation('sigmoid')

        # position
        x_pos = self.net_position(x)
        
        # orientation
        x_ori = self.net_orientation(x)
        x_ori = F.normalize(x_ori, p=2, dim=1)

        # size
        x_size = self.net_size(x)
        x_size = 0.5 * sigmoid(x_size) + 0.03
        # x_size_a1 = 0.1 * sigmoid(x_size[:, 0:1]) + 0.03
        # x_size_a2 = 0.1 * sigmoid(x_size[:, 1:2]) + 0.03
        # x_size_a3 = 0.5 * sigmoid(x_size[:, 2:3]) + 0.03
        # x_size = torch.cat([x_size_a1, x_size_a2, x_size_a3], dim=1)

        # shape
        x_shape = self.net_shape(x)
        x_shape = 1.5 * sigmoid(x_shape) + 0.2

        # # slice
        # x_slice = self.net_slice(x)
        # x_slice_c1 = 1.4 * sigmoid(x_slice[:, :1]) - 1.0
        # x_slice_c2 = 0.5 * sigmoid(x_slice[:, 1:]) + 0.5
        # x_slice = torch.cat([x_slice_c1, x_slice_c2], dim=1)

        x_cat = torch.cat([x_pos, x_ori, x_size, x_shape], dim=1)

        # taper
        if 'taper' in self.module_collection.keys():
            x_taper = self.net_taper(x)
            x_taper = 1.8 * sigmoid(x_taper) - 0.9
            x_cat = torch.cat([x_cat, x_taper], dim=1)

        # bending
        if 'bending' in self.module_collection.keys():
            x_bend = self.net_bend(x)
            # x_bend_k = 0.1 + 7.9 * sigmoid(x_bend[:, :1])
            x_bend_k = 0.01 + 0.74 * sigmoid(x_bend[:, :1])
            # x_bend_k = - 0.06 - 0.74 * sigmoid(x_bend[:, :1])
            # x_bend_k = 1.0 + leakyrelu(x_bend[:, :1])
            # x_bend_a = 0 * x_bend[:, 1:]
            # x_bend_a = F.normalize(x_bend[:, 1:], p=2, dim=1)

            x_bend_a = self.net_bend_angle(x)
            x_bend_a = F.normalize(x_bend_a, p=2, dim=1)
            x_cat = torch.cat([x_cat, x_bend_k, x_bend_a], dim=1)

        return x_cat

class Estimator_membership(nn.Module):
    def __init__(self, **args):
        super(Estimator_membership, self).__init__()

    def forward(self, x):
        # x : (batch, n_primitives(membership), num_point(3000))
        softmax = get_activation('softmax')
        x = softmax(x)
        return x.transpose(1, 2)