import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy 

class BaseArch(nn.Module):

    def __init__(self, backbone, head_model, use_label = False, use_superquadric_label = False):
        super(BaseArch, self).__init__()
        self.backbone = backbone
        self.module_net = nn.Sequential(*head_model)
        self.use_label = use_label
        self.use_superquadric_label = use_superquadric_label

    # def model_initialize(self, train_loader, device):
    #     delta_rot_average = torch.zeros(1, 3,3).to(device)
    #     n = 0
    #     for x, x_gt, y in train_loader:
    #         output = self(x.to(device))
    #         quat = output[:, 3:7]
    #         rot = lie_torch.quaternions_to_rotation_matrices_torch(quat)
    #         rot_esti = y[:, :, 4:13].to(device).view(-1,3,3)
    #         rot_copied = copy.copy(rot)
    #         delta_rot_average = delta_rot_average + lie_torch.log_SO3(rot_copied.permute(0, 2, 1)@rot_esti)
    #         n = n + 1
    #         # if n == 10:
    #         #     break
    #     delta_rot_average = lie_torch.exp_so3(delta_rot_average / n)
    #     self.offset_SO3 = delta_rot_average
    #     print('offset SO3 is initialized by: {}'.format(self.offset_SO3))

    def forward(self, x):

        if self.backbone.output_feature == 'local':
            x = self.backbone.local_feature_map(x)
        elif self.backbone.output_feature == 'global':
            x = self.backbone.global_feature_map(x)
        elif self.backbone.output_feature == 'local_global':
            x = self.backbone.local_global_feature_map(x)
        else:
            raise ("Specify output feature of backbone: local, global, local_global")

        x = self.module_net(x)

        return x

        # if hasattr(self, 'offset_SO3'):
        #     rot_prev = lie_torch.quaternions_to_rotation_matrices_torch(x[:, 3:7])
        #     rot = rot_prev@copy.copy(self.offset_SO3.detach())
        #     modified_quat = lie_torch.rotation_matrices_to_quaternions_torch(rot)

        #     return torch.cat((x[:, :3], modified_quat, x[:, 7:]), dim=1)
        # else:
        #     return x

    def train_step(self, x, y, optimizer, loss, clip_grad=1, x_gt = None, l_gt = None, **kwargs):
        optimizer.zero_grad()
        if self.use_superquadric_label is False:
            if self.use_label is True:
                loss_ = loss(y, self(x))
            elif self.use_label is False:
                if x_gt is not None:
                    loss_ = loss(x_gt, self(x))
                else:
                    loss_ = loss(x, self(x))
            else:
                raise ("Check the use_label dictionary component: True or False")
        elif self.use_superquadric_label is True:
            loss_ = loss(x_gt, l_gt, self(x))
        else:
            raise ("Check the use_superquadric_label dictionary component: True or False")
        loss_.backward(retain_graph=True)
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()

        # input point cloud
        x_pc = x.detach().cpu().permute([0,2,1]).numpy()
        # x_pc = x_gt.detach().cpu().permute([0,2,1]).numpy()

        # input ground truth points
        x_pc_gt = x_gt[:,:3,:].detach().cpu().permute([0,2,1]).numpy()
        x_l_gt = x_gt[:,6:,:].detach().cpu().permute([0,2,1]).numpy()
                
        # ground truth primitives
        y_gt = y.detach().cpu().numpy()
        
        # fitted primitives
        y_f = self(x).detach().cpu().numpy()

        return {'loss': loss_.item(), 
                'pointcloud@': x_pc,
                # 'gtpointcloudwboundary!': [x_pc_gt, x_l_gt],
                'gtpointcloud)': x_pc_gt,
                'diffcolor#': [x_pc, y_gt, y_f]
                }

    def validation_step(self, x, y, loss, x_gt = None, l_gt = None, **kwargs):

        if self.use_superquadric_label is False:
            if self.use_label is True:
                loss_ = loss(y, self(x))
            elif self.use_label is False:
                if x_gt is not None:
                    loss_ = loss(x_gt, self(x))
                else:
                    loss_ = loss(x, self(x))
            else:
                raise ("Check the use_label dictionary component: True or False")
        elif self.use_superquadric_label is True:
            loss_ = loss(x_gt, l_gt, self(x))
        else:
            raise ("Check the use_superquadric_label dictionary component: True or False")

        # input point cloud
        x_pc = x.detach().cpu().permute([0,2,1]).numpy()
        # x_pc = x_gt.detach().cpu().permute([0,2,1]).numpy()
        
        # input ground truth points
        x_pc_gt = x_gt[:,:3,:].detach().cpu().permute([0,2,1]).numpy()

        # ground truth primitives
        y_gt = y.detach().cpu().numpy()
        
        # fitted primitives
        y_f = self(x).detach().cpu().numpy()

        return {'loss': loss_.item(), 
                'pointcloudval@': x_pc,
                # 'gtpointcloudwboundary!': [x_pc_gt, x_l_gt],
                'gtpointcloud)': x_pc_gt,
                'diffcolorval#': [x_pc, y_gt, y_f]
                }


