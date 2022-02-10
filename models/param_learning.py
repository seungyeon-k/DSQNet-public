import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.functions.lie_torch import exp_se3

class ParamLearning(nn.Module):
    def __init__(self, **kargs):
        super(ParamLearning, self).__init__()
        self.kargs = kargs
        self.emb_dims = kargs['emb_dims']
        self.n_primitives = kargs['n_primitives']
        self.list_num_each_param = kargs['list_num_each_param']
        self.n_types = kargs['n_types']
        self.n_parameters = sum(self.list_num_each_param)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, kargs['emb_dims'], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(kargs['emb_dims'])
        self.linear1 = nn.Linear(kargs['emb_dims'], 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, self.n_primitives * self.n_parameters)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x).view(-1, self.n_primitives, self.n_parameters)
        # x[:, :, 0:4] = self.softmax(x[:, :, 0:4])

        x_type = x[:, :, :self.n_types]
        x_SE3 = x[:, :, self.n_types:self.n_types + 6]
        x_SE3 = exp_se3(x_SE3.view(-1, 6))
        x_SE3 = x_SE3.view(-1, self.n_primitives, 4, 4)
        
        # x_SE3 = x_SE3[:, :, :3, :].view(-1, self.n_primitives, 12)
        x_SO3 = x_SE3[:, :, :3, :3].reshape(-1, self.n_primitives, 9)
        x_p = x_SE3[:, :, :3, 3:].reshape(-1, self.n_primitives, 3)

        x_physical_params = x[..., self.n_types + 12:]
        offset = 1e-05
        x_physical_params = F.relu(x_physical_params) + offset
        x_type = self.softmax(x_type)

        return torch.cat((x_type, x_SO3, x_p, x_physical_params), dim=2)

    def train_step(self, x, y, optimizer, loss, clip_grad=None, **kwargs):
        optimizer.zero_grad()
        loss_ = loss(y, self(x))
        loss_.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()

        # input point cloud
        x_pc = x.detach().cpu().permute([0,2,1]).numpy()
        # x_pc_norm = np.zeros(np.shape(x_pc)) # normalization is needed for add_mesh
        # for i in range(np.shape(x_pc)[0]):
        #     max_element = np.max(x_pc[i,:,:])
        #     min_element = np.min(x_pc[i,:,:])
        #     x_pc_norm[i,:,:] = (2 * x_pc[i,:,:] - (max_element + min_element)) / (max_element - min_element)
        
        # ground truth primitives
        y_gt_primitive = y.detach().cpu().numpy()
        
        # fitted primitives
        y_f_primitive = self(x).detach().cpu().numpy()

        return {'loss': loss_.item(), 
                'pointcloud@': x_pc,
                'groundtruth$': y_gt_primitive,  
                'trained%': y_f_primitive,
                'gtandtrained*': [y_gt_primitive, y_f_primitive]}

    def validation_step(self, x, y, loss, **kwargs):
        loss_ = loss(y, self(x))

        # if kwargs.get('show_pointcloud', True):
        #     x_img = make_grid(x.detach().cpu(), nrow=10, range=(0, 1))
        #     recon_img = make_grid(recon.detach().cpu(), nrow=10, range=(0, 1))
        # else:
        #     x_img, recon_img = None, None
        return {'loss': loss_.item()}#, 'predict': predict, 'reconstruction': recon,
                #'input@': x_img, 'recon@': recon_img}