import numpy as np
import open3d as o3d
import os
import pandas
import sys
import torch
from torch.utils.data.sampler import SubsetRandomSampler
sys.path.append("..")
from functions import lie
from sklearn.preprocessing import normalize

class PrimitiveDataset(torch.utils.data.Dataset): 

    def __init__(self, split, data_cfg):
                
        # data path
        data_path = data_cfg["path"]

        # csv file
        csv_path = data_path + '/' + data_cfg["csv_name"]
        csv_file = pandas.read_csv(csv_path, delimiter=',', header=None)
        data_namelist = list(csv_file[0])

        # initialization
        object_types = data_cfg['object_types']
        n_types = data_cfg['n_types']
        info_types = data_cfg['info_types']
        n_params = data_cfg['n_params']
        n_primitives = data_cfg['n_primitives']
        full_num_params = data_cfg['full_num_params']
        num_pointclouds = data_cfg['num_pointclouds']
        self.num_pointclouds = num_pointclouds
        self.num_pointcloudsinput = data_cfg['num_pointcloudsinput']
        num_gt_pointclouds = data_cfg['num_gt_pointclouds']
        input_normalization = data_cfg['input_normalization']
        full_types = full_num_params.keys()

        pc_list = []
        pc_gt_list = []
        pc_gt_split_list = []
        label_list = []
        superquadric_label_list = []
        for file in data_namelist:

            # data name processing
            underbar_index = [index for index, char in enumerate(file) if char == '_']
            dot_index = [index for index, char in enumerate(file) if char == '.']
            npy_index = dot_index[0]
            prefix_index = underbar_index[-3]
            file_prefix = file[0:prefix_index]
            viewpoint_index = underbar_index[-1]
            viewpoint = file[viewpoint_index+1:npy_index]
    
            # interested objects
            if file_prefix not in object_types:
                continue
            
            # top-down view point exclusion
            if viewpoint == '00' or viewpoint == '15':
                continue

            # data load
            data = np.load(data_path + '/' + file_prefix + '/' + file, allow_pickle = True).item()

            if not np.shape(data["partial_pc"])[0] == num_pointclouds:
                print('The number of partial pointclouds of {} is not {}'.format(file, str(num_pointclouds)))
                continue
            if not np.shape(data["full_pc"])[0] == num_gt_pointclouds:
                print('The number of full pointclouds of {} is not {}'.format(file, str(num_gt_pointclouds)))
                continue

            # point cloud data
            pc = data["partial_pc"].transpose()
            noise = np.random.uniform(-1, 1, size=pc.shape)
            noise = normalize(noise, axis=0, norm='l2')
            noise_std = 0.001
            scale = np.random.normal(loc=0, scale=noise_std, size=(1, pc.shape[1])).repeat(pc.shape[0], axis=0)
            pc = pc + noise * scale
            pc_list.append(pc)

            # ground truth point cloud data
            pc_gt = data["full_pc"]
            pc_gt_list.append(pc_gt.transpose())

            # only for SINGLE primitives 
            if len(data["primitives"]) is not 1:
                raise ValueError('primitives are more than 1') 
            primitive = data["primitives"][0]

            # total label list
            label = []
            
            # type vector
            label_type = [0.] * n_types
            index = 0
            for type_ in full_types:
                if type_ in info_types:
                    if primitive['type'] == type_:
                        label_type[index] = 1.
                    index += 1
            
            # calculate position and orientation
            pose = primitive['SE3']
            label_so3 = pose[0,:3].tolist() + pose[1,:3].tolist() + pose[2,:3].tolist()
            label_p = pose[0,3:].tolist() + pose[1,3:].tolist() + pose[2,3:].tolist()

            # parameter vector
            parameters = primitive['parameters']	
            label_params = [0.] * n_params
            index = 0
            for type_ in full_types:
                if type_ in info_types:
                    if primitive['type'] == type_:
                        for _, value in parameters.items():
                            label_params[index] = value
                            index = index + 1

                        # for calculating boundary
                        boundary_height = value
                        # if primitive['type'] == 'truncated_torus':
                        #     boundary_height = label_params[index-3] + label_params[index-2] - label_params[index-1]
                    else:
                        index += full_num_params[type_]

            # append
            label.append(label_type + label_p + label_so3 + label_params + [1])
            label_list.append(label)

            # print(file_prefix)
            # print(label)

            # superquadric label_list
            superquadric_label = orientation_prior(file_prefix, pose)
            # superquadric_label = shape_prior(file_prefix, parameters)
            superquadric_label_list.append(superquadric_label)

            # # boundary-aware
            # pc_gt_homo = np.concatenate((pc_gt[:, :3].transpose(), np.ones((1, pc_gt.shape[0]))), axis=0)
            # pc_gt_transformed = np.linalg.inv(pose).dot(pc_gt_homo)

            # pc_upper_index = pc_gt_transformed[2, :] > boundary_height / 2 - 1e-4
            # pc_lower_index = pc_gt_transformed[2, :] < -boundary_height / 2 + 1e-4
            # pc_inner_index = np.logical_and(~pc_upper_index, ~pc_lower_index)
            # pc_gt_upper = pc_gt[pc_upper_index, :]
            # pc_gt_lower = pc_gt[pc_lower_index, :]
            # pc_gt_inner = pc_gt[pc_inner_index, :]
            # pc_gt_upper = np.concatenate((pc_gt_upper, 1*np.ones((pc_gt_upper.shape[0], 1))), axis=1)
            # pc_gt_lower = np.concatenate((pc_gt_lower, 2*np.ones((pc_gt_lower.shape[0], 1))), axis=1)
            # pc_gt_inner = np.concatenate((pc_gt_inner, 3*np.ones((pc_gt_inner.shape[0], 1))), axis=1)
            # pc_gt_split = np.concatenate((pc_gt_upper.transpose(), pc_gt_lower.transpose(), pc_gt_inner.transpose()), axis=1)
            # pc_gt_split_list.append(pc_gt_split)

            # if not (sum(pc_upper_index) + sum(pc_lower_index) + sum(pc_inner_index) == 1500):
            #     print(boundary_height)

            # if file_prefix == 'box':
            #     print(file_prefix)
            #     print(pc_gt_upper.shape, pc_gt_inner.shape, pc_gt_lower.shape)

            # if file_prefix == 'cylinder':
            #     print(file_prefix)
            #     print(pc_gt_upper.shape, pc_gt_inner.shape, pc_gt_lower.shape)

            # if file_prefix == 'truncated_cone':
            #     pc_upper_index = pc_gt_transformed[2, :] > boundary_height / 2 - 1e-5
            #     pc_lower_index = pc_gt_transformed[2, :] < -boundary_height / 2 + 1e-5
            #     pc_inner_index = np.logical_and(~pc_upper_index, ~pc_lower_index)
            #     pc_gt_upper = pc_gt[pc_upper_index, :]
            #     pc_gt_lower = pc_gt[pc_lower_index, :]
            #     pc_gt_inner = pc_gt[pc_inner_index, :]
            #     pc_gt_upper = np.concatenate((pc_gt_upper, 1*np.ones((pc_gt_upper.shape[0], 1))), axis=1)
            #     pc_gt_lower = np.concatenate((pc_gt_lower, 2*np.ones((pc_gt_lower.shape[0], 1))), axis=1)
            #     pc_gt_inner = np.concatenate((pc_gt_inner, 3*np.ones((pc_gt_inner.shape[0], 1))), axis=1)
            #     pc_gt_split = np.concatenate((pc_gt_upper.transpose(), pc_gt_lower.transpose(), pc_gt_inner.transpose()), axis=1)
            #     pc_gt_split_list.append(pc_gt_split)
            
            # elif file_prefix == 'cone':
            #     pc_upper_index = pc_gt_transformed[2, :] > boundary_height / 2 - 1e-5
            #     pc_lower_index = pc_gt_transformed[2, :] < -boundary_height / 2 + 1e-5
            #     pc_inner_index = np.logical_and(~pc_upper_index, ~pc_lower_index)
            #     pc_gt_upper = pc_gt[pc_upper_index, :]
            #     pc_gt_lower = pc_gt[pc_lower_index, :]
            #     pc_gt_inner = pc_gt[pc_inner_index, :]
            #     pc_gt_upper = np.concatenate((pc_gt_upper, 1*np.ones((pc_gt_upper.shape[0], 1))), axis=1)
            #     pc_gt_lower = np.concatenate((pc_gt_lower, 2*np.ones((pc_gt_lower.shape[0], 1))), axis=1)
            #     pc_gt_inner = np.concatenate((pc_gt_inner, 3*np.ones((pc_gt_inner.shape[0], 1))), axis=1)
            #     pc_gt_split = np.concatenate((pc_gt_upper.transpose(), pc_gt_lower.transpose(), pc_gt_inner.transpose()), axis=1)
            #     pc_gt_split_list.append(pc_gt_split)

            # elif file_prefix == 'truncated_sphere':
            #     pc_upper_index = pc_gt_transformed[2, :] > boundary_height / 2 - 1e-5
            #     pc_lower_index = pc_gt_transformed[2, :] < -boundary_height / 2 + 1e-5
            #     pc_inner_index = np.logical_and(~pc_upper_index, ~pc_lower_index)
            #     pc_gt_upper = pc_gt[pc_upper_index, :]
            #     pc_gt_lower = pc_gt[pc_lower_index, :]
            #     pc_gt_inner = pc_gt[pc_inner_index, :]
            #     pc_gt_upper = np.concatenate((pc_gt_upper, 1*np.ones((pc_gt_upper.shape[0], 1))), axis=1)
            #     pc_gt_lower = np.concatenate((pc_gt_lower, 2*np.ones((pc_gt_lower.shape[0], 1))), axis=1)
            #     pc_gt_inner = np.concatenate((pc_gt_inner, 3*np.ones((pc_gt_inner.shape[0], 1))), axis=1)
            #     pc_gt_split = np.concatenate((pc_gt_upper.transpose(), pc_gt_lower.transpose(), pc_gt_inner.transpose()), axis=1)
            #     pc_gt_split_list.append(pc_gt_split)

            # elif file_prefix == 'truncated_torus':
            #     pc_side_index = pc_gt_transformed[0, :] > boundary_height - 1e-5
            #     pc_inner_index = ~pc_side_index
            #     pc_gt_side = pc_gt[pc_side_index, :]
            #     pc_gt_inner = pc_gt[pc_inner_index, :]
            #     pc_gt_side = np.concatenate((pc_gt_side, 1*np.ones((pc_gt_side.shape[0], 1))), axis=1)
            #     pc_gt_inner = np.concatenate((pc_gt_inner, 3*np.ones((pc_gt_inner.shape[0], 1))), axis=1)
            #     pc_gt_split = np.concatenate((pc_gt_side.transpose(), pc_gt_inner.transpose()), axis=1)
            #     pc_gt_split_list.append(pc_gt_split)

            # else:
            #     pc_gt_inner = np.concatenate((pc_gt, 3*np.ones((pc_gt.shape[0], 1))), axis=1)
            #     pc_gt_split = pc_gt_inner.transpose()
            #     pc_gt_split_list.append(pc_gt_split)

        # convert to numpy
        batch_size = len(pc_list)
        pc_list_numpy = np.array(pc_list)
        pc_gt_list_numpy = np.array(pc_gt_list)
        # pc_gt_split_list_numpy = np.array(pc_gt_split_list)
        label_list_numpy = np.array(label_list)
        superquadric_label_list_numpy = np.array(superquadric_label_list)
        self.normalizer = None

        if input_normalization:
            # point cloud normalization
            max_ = np.max(pc_list_numpy, axis=2)
            min_ = np.min(pc_list_numpy, axis=2)
            diagonal_len = np.linalg.norm(max_-min_, axis=1)
            pc_list_numpy = pc_list_numpy / diagonal_len.reshape([batch_size, 1, 1])
            pc_gt_list_numpy[:,:3,:] = pc_gt_list_numpy[:,:3,:] / diagonal_len.reshape([batch_size, 1, 1])
            # pc_gt_split_list_numpy[:,:3,:] = pc_gt_split_list_numpy[:,:3,:] / diagonal_len.reshape([batch_size, 1, 1])
            label_list_numpy[:,:,n_types:n_types+3] = label_list_numpy[:,:,n_types:n_types+3] / diagonal_len.reshape([batch_size, 1, 1])
            # label_list_numpy[:,:,n_types+12:n_types+n_params+12] = label_list_numpy[:,:,n_types+12:n_types+n_params+12] / diagonal_len.reshape([batch_size, 1, 1])
            label_list_numpy[:,:,n_types+12:n_types+n_params+12 - 5] = label_list_numpy[:,:,n_types+12:n_types+n_params+12 - 5] / diagonal_len.reshape([batch_size, 1, 1])
            label_list_numpy[:,:,n_types+n_params+12- 3:n_types+n_params+12 - 1] = label_list_numpy[:,:,n_types+n_params+12- 3:n_types+n_params+12 - 1] / diagonal_len.reshape([batch_size, 1, 1])
            self.normalizer = diagonal_len
            
        self.pc_list = pc_list_numpy.tolist()
        # self.pc_gt_list = pc_gt_split_list_numpy.tolist()
        self.pc_gt_list = pc_gt_list_numpy.tolist()
        self.label_list = label_list_numpy.tolist()
        self.superquadric_label_list = superquadric_label_list_numpy.tolist()

                           
    def __len__(self):
        return len(self.pc_list)

    def __getitem__(self, idx): 

        # random idx
        perm = torch.randperm(self.num_pointclouds)
        permidx = perm[:self.num_pointcloudsinput]
        tempidx = torch.zeros(self.num_pointclouds)
        tempidx[permidx] = 1
        tempidx = tempidx.type(torch.bool)
        x = torch.Tensor(self.pc_list[idx])[:, tempidx]

        x_gt = torch.Tensor(self.pc_gt_list[idx])
        y = torch.Tensor(self.label_list[idx])
        l_gt = torch.Tensor(self.superquadric_label_list[idx])
        return x, x_gt, y, l_gt

def shape_prior(primitive_type, dict_parameters):

    if primitive_type == 'box':
        superquadric_label = [0, 0.2, 0.2]
    elif primitive_type == 'sphere':
        superquadric_label = [0, 1, 1]
    elif primitive_type == 'truncated_sphere':
        superquadric_label = [0, 1, 1] 
    elif primitive_type == 'cylinder':
        superquadric_label = [0, 0.2, 1]
    elif primitive_type == 'cone':
        superquadric_label = [0, 2, 1]
    elif primitive_type == 'truncated_cone':
        superquadric_label = [0, 2, 1]    
    elif primitive_type == 'torus':
        superquadric_label = [dict_parameters['torus_radius'] / dict_parameters['tube_radius'], 1, 1]
    elif primitive_type == 'truncated_torus':
        superquadric_label = [dict_parameters['torus_radius'] / dict_parameters['tube_radius'], 1, 1]
    elif primitive_type == 'rectangle_ring':
        superquadric_label = [(dict_parameters['depth'] - dict_parameters['thickness2']) / dict_parameters['thickness2'], 0.2, 0.2]
    elif primitive_type == 'truncated_rectangle_ring':
        superquadric_label = [(dict_parameters['depth'] - dict_parameters['thickness2']) / dict_parameters['thickness2'], 0.2, 0.2]
    elif primitive_type == 'cylinder_ring':
        superquadric_label = [2 * dict_parameters['torus_radius'] / (dict_parameters['radius_outer'] - dict_parameters['radius_inner']) - 1, 0.2, 1]
    elif primitive_type == 'truncated_cylinder_ring':
        superquadric_label = [2 * dict_parameters['torus_radius'] / (dict_parameters['radius_outer'] - dict_parameters['radius_inner']) - 1, 0.2, 1]
    elif primitive_type == 'semi_sphere_shell':
        superquadric_label = [1110, 1111, 1101]
    else:
        raise ValueError('primitive type is not understood') 

    return superquadric_label

def orientation_prior(primitive_type, pose):

    if primitive_type == 'truncated_torus':
        z = pose[:3,1].tolist()
    else:
        z = pose[:3,2].tolist()

    return z

    # y = pose[:3,1].tolist()

    # return y