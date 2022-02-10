import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseArchSeg(nn.Module):

    def __init__(self, backbone, head_model):
        super(BaseArchSeg, self).__init__()
        self.backbone = backbone
        self.module_net = nn.Sequential(*head_model)

    def forward(self, x):
        x = self.backbone.local_global_feature_map(x)
        x = self.module_net(x)
        return x

    def train_step(self, x, y, optimizer, loss, clip_grad=None, **kwargs):
        optimizer.zero_grad()
        loss_ = loss(y, self(x))
        loss_.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
        optimizer.step()

        # input point cloud
        x_pc = x.detach().cpu().permute([0,2,1]).numpy()
        
        # ground truth primitives
        y_gt = y.detach().cpu().numpy()
        
        # fitted primitives
        y_f = self(x).detach().cpu().numpy()

        return {'loss': loss_.item(), 
                'pointcloud': x_pc,
                'data+': [x_pc, y_gt, y_f]
                }

    def validation_step(self, x, y, loss, **kwargs):
        loss_ = loss(y, self(x))

        # input point cloud
        x_pc = x.detach().cpu().permute([0,2,1]).numpy()
        
        # ground truth primitives
        y_gt = y.detach().cpu().numpy()
        
        # fitted primitives
        y_f = self(x).detach().cpu().numpy()

        return {'loss': loss_.item(), 
                'pointcloud': x_pc,
                'data+': [x_pc, y_gt, y_f]
                }