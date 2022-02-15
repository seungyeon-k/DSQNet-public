import numpy as np
import pandas
import torch
from sklearn.preprocessing import normalize

class PrimitiveDataset(torch.utils.data.Dataset): 

    def __init__(self, data_cfg):
                
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
            pc_gt = data["full_pc"].transpose()
            pc_gt_list.append(pc_gt)

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
                    else:
                        index += full_num_params[type_]

            # append
            label.append(label_type + label_p + label_so3 + label_params + [1])
            label_list.append(label)

            # superquadric label_list
            superquadric_label = orientation_prior(file_prefix, pose)
            superquadric_label_list.append(superquadric_label)

        # convert to numpy
        batch_size = len(pc_list)
        pc_list_numpy = np.array(pc_list)
        pc_gt_list_numpy = np.array(pc_gt_list)
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


def orientation_prior(primitive_type, pose):

    if primitive_type == 'truncated_torus':
        z = pose[:3,1].tolist()
    else:
        z = pose[:3,2].tolist()
    return z
