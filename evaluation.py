import numpy as np
import os
import argparse
import torch
import open3d as o3d
from datetime import datetime
from copy import deepcopy
from sklearn.preprocessing import normalize
from tensorboardX import SummaryWriter

from functions.pc_preprocess import process_pc, normalize_numpy_pc
from functions.utils_numpy import quaternions_to_rotation_matrices, define_SE3, inverse_SE3
from functions.primitives import Superquadric, DeformableSuperquadric
from functions.object_class import Object
from functions.data_reader import create_object_from_data
from functions.iou_calculator import iou
from models import load_pretrained

def pred_to_parameters(pred_fit, normalizer, pnts_frame):

    # SE3
    SE3 = define_SE3(np.squeeze(quaternions_to_rotation_matrices(np.expand_dims(pred_fit[3:7], axis=0))), normalizer * pred_fit[:3])
    SE3 = np.dot(pnts_frame, SE3)  
    
    # parameters
    parameters = dict()
    parameters['a1'] = pred_fit[7] * normalizer
    parameters['a2'] = pred_fit[8] * normalizer
    parameters['a3'] = pred_fit[9] * normalizer
    parameters['e1'] = pred_fit[10]
    parameters['e2'] = pred_fit[11]
    if len(pred_fit) > 12:
        parameters['k'] = pred_fit[12]
        parameters['b'] = pred_fit[13]
        parameters['cos_alpha'] = pred_fit[14]
        parameters['sin_alpha'] = pred_fit[15]

    return SE3, parameters

if __name__ == '__main__':

    #########################################################################
    ########################### Initial Settings ############################
    #########################################################################

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--device', default=0)
    parser.add_argument('--run', default=None)
    parser.add_argument('--iou', action='store_true')
    args = parser.parse_args()

    # configuration
    device = f'cuda:{args.device}'
    logdir_home = 'evaluation_results/tensorboard'
    if args.run is not None:
        run_id = args.run
    else:
        run_id = datetime.now().strftime('%Y%m%d-%H%M')
    example_object = args.object
    example_idx = args.index

    # pre-trained networks
    seg_config = ('segnet_config/example', 'segnet_config.yml', 'model_best.pkl', {})
    sq_config = ('sqnet_config/example', 'sqnet_config.yml', 'model_best.pkl', {})
    dsq_config = ('dsqnet_config/example', 'dsqnet_config.yml', 'model_best.pkl', {})
    root = 'pretrained/'

    # evaluation dataset configuration
    data_path = 'datasets/evaluation_dataset'

    # parameters
    num_processed_pc_points = 1000
    num_fitting_pc_points = 300

    # save path
    save_folder = 'evaluation_results/data'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)   
    save_name = os.path.join(save_folder, str(run_id))

    # color parameters
    prim_clr_dict = [
        [0, 98, 156], 
        [199, 202, 86], 
        [208, 45, 35], 
        [255, 106, 110],
        [177, 39, 99],
        [255, 173, 79]
    ]
    seg_clr_dict = [
        [129, 228, 189],
        [251, 208, 210],
        [242, 230, 81],
        [114, 47, 55],
        [101, 141, 198],
        [248, 212, 152]
    ]
    pnt_clr = [83, 86, 90]

    #########################################################################
    #################### Load Pre-trained Models ############################
    #########################################################################

    # load configuration 
    seg_identifier, seg_config_file, seg_ckpt_file, seg_kwargs = seg_config
    sq_identifier, sq_config_file, sq_ckpt_file, sq_kwargs = sq_config
    dsq_identifier, dsq_config_file, dsq_ckpt_file, dsq_kwargs = dsq_config

    # Load pretrained model
    seg_model, _ = load_pretrained(seg_identifier, seg_config_file, seg_ckpt_file, root=root, **seg_kwargs)
    sqnet_model, _ = load_pretrained(sq_identifier, sq_config_file, sq_ckpt_file, root=root, **sq_kwargs)
    dsqnet_model, _ = load_pretrained(dsq_identifier, dsq_config_file, dsq_ckpt_file, root=root, **dsq_kwargs)
    seg_model.to(device)
    sqnet_model.to(device)
    dsqnet_model.to(device)

    #########################################################################
    ############################# Evaluation ################################
    #########################################################################

    # object number
    file_number = f"{example_idx + 90:04}"
    viewpoint = np.random.choice(14, size=14, replace = False)

    viewnum = 0
    breaknum = 0
    while True:
        
        # terminate
        if breaknum == 3:
            break

        # viewpoint
        file_viewpoint = f"{viewpoint[viewnum] + 1:02}"
        file_name = f'{example_object}_{file_number}_viewpoint_{file_viewpoint}'

        # load pointcloud
        data_name = os.path.join(data_path, example_object, f'{file_name}.npy')
        if os.path.exists(data_name):
            data = np.load(data_name, allow_pickle = True).item()
            viewnum = viewnum + 1
            breaknum = breaknum + 1
        else:
            viewnum = viewnum + 1
            continue
        pointcloud = data["partial_pc"]
        membership = data["membership"]
        gt_membership_type = []
        for primitive in data['primitives']:
            type_ = primitive['type']
            if type_ == 'cone' and primitive['parameters']['upper_radius'] > 0:
                type_ = 'truncated_cone'
            gt_membership_type.append(type_)

        # open tensorboard
        logdir = os.path.join(logdir_home, str(run_id), file_name)
        writer = SummaryWriter(logdir=logdir)
        print("Result directory: {}".format(logdir))

        # noise
        pointcloud = pointcloud.transpose()
        noise = np.random.uniform(-1, 1, size=pointcloud.shape)
        noise = normalize(noise, axis=0, norm='l2')
        noise_std = 0.001
        scale = np.random.normal(loc=0, scale=noise_std, size=(1, pointcloud.shape[1])).repeat(pointcloud.shape[0], axis=0)
        pointcloud = pointcloud + noise * scale
        pointcloud = pointcloud.transpose()

        # random idx
        perm = np.random.permutation(3000)
        permidx = perm[:1000]
        tempidx = np.zeros(3000, dtype=np.bool_)
        tempidx[permidx] = True
        pointcloud = pointcloud[tempidx, :]
        membership = membership[tempidx]

        # ground truth mesh
        gt_mesh = create_object_from_data(data)
        vertices_gt = np.expand_dims(np.asarray(gt_mesh.mesh.vertices), axis=0)
        faces_gt = np.expand_dims(np.asarray(gt_mesh.mesh.triangles), axis=0)   
        writer.add_mesh('groundtruth_object', vertices=vertices_gt, faces=faces_gt, global_step=1
        )

        # segment partially observed point cloud
        pcd = o3d.geometry.PointCloud(points = o3d.utility.Vector3dVector(pointcloud[:,0:3]))
        pcd, pcd_frame = process_pc(pcd, num_processed_pc_points=num_processed_pc_points)
        pcd_np_normalized, normalizer_seg = normalize_numpy_pc(np.asarray(pcd.points))
        pcd.points = o3d.utility.Vector3dVector(pcd_np_normalized)
        pnts_tensor = torch.tensor(np.expand_dims(np.array(pcd.points).transpose(), axis=0) , dtype=torch.float).to(device)
        pred = seg_model(pnts_tensor)
        pcd_seg = np.argmax(np.squeeze(pred.detach().cpu().numpy()), axis=1)
        
        # seperate segmented pnts to seperate list
        pnts = np.asarray(pcd.points)
        pnts = pnts * normalizer_seg
        seg_pnts_list = []
        for p in range(max(pcd_seg)+1):
            seg_pnts = pnts[np.squeeze(np.argwhere(np.array(pcd_seg) == p)), :]
            if len(seg_pnts) > 100:
                seg_pnts_list.append(seg_pnts)
        seg_pnts_list_prev = deepcopy(seg_pnts_list)
        for p, seg_pnts in enumerate(seg_pnts_list):
            seg_pnts_list_prev[p] = np.dot(np.concatenate((seg_pnts_list_prev[p], np.ones((seg_pnts_list_prev[p].shape[0], 1))), axis=1), inverse_SE3(pcd_frame).transpose())

        # segmentated pointcloud
        seg_color = np.zeros((len(pcd.points), 3))
        random_segclr_idx = np.random.choice(6, size=max(pcd_seg)+1, replace=False)
        for pnt in range(len(pointcloud)):
            seg_color[pnt, :] = seg_clr_dict[random_segclr_idx[pcd_seg[pnt]]]
        pcd.colors = o3d.utility.Vector3dVector(seg_color / 255)
        max_bound = pcd.get_max_bound()
        min_bound = pcd.get_min_bound()
        resolution = 20
        processed_pcd_ds = pcd.voxel_down_sample(np.max(max_bound-min_bound) / resolution)
        seg_color = np.asarray(processed_pcd_ds.colors) * 255
        pointcloud_batch = np.expand_dims(np.asarray(processed_pcd_ds.points), axis=0)
        segcolor_batch = np.expand_dims(seg_color, axis=0)
        writer.add_mesh(
            'segmentation_results', 
            vertices=pointcloud_batch, 
            colors = segcolor_batch, 
            global_step=1
        )

        # draw pointcloud
        processed_pcd_ds.paint_uniform_color(np.array(pnt_clr) / 255)
        input_color = np.asarray(processed_pcd_ds.colors) * 255
        input_color_batch = np.expand_dims(input_color, axis=0)
        writer.add_mesh(
            'observed_point_cloud', 
            vertices=pointcloud_batch,
            colors = input_color_batch, 
            global_step=1
        )

        # process segmented pc (upsampling & frame transform)
        for p, seg_pnts in enumerate(seg_pnts_list):
            seg_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(seg_pnts))
            seg_pcd, seg_pcd_frame = process_pc(seg_pcd, num_processed_pc_points=num_fitting_pc_points)
            seg_pnts = np.asarray(seg_pcd.points)
            seg_pnts_list[p] = (seg_pnts, seg_pcd_frame)
        
        # shape primitive recognition for sqnet and dsqnet
        sq_set = []
        dsq_set = []
        for p, seg_info in enumerate(seg_pnts_list):
            seg_pnts, normalizer = normalize_numpy_pc(seg_info[0])
            
            # sqnet
            pnts_tensor = torch.tensor(np.expand_dims(seg_pnts.transpose(), axis=0) , dtype=torch.float).to(device)
            pred_fit = sqnet_model(pnts_tensor)
            pred_fit = np.squeeze(pred_fit.detach().cpu().numpy())
            sq_SE3, sq_parameters = pred_to_parameters(pred_fit, normalizer, seg_info[1])
            sq_SE3 = np.dot(pcd_frame, sq_SE3)
            sq_fit = Superquadric(sq_SE3, sq_parameters)
            sq_set.append(sq_fit)

            # dsqnet
            pnts_tensor = torch.tensor(np.expand_dims(seg_pnts.transpose(), axis=0) , dtype=torch.float).to(device)
            pred_fit = dsqnet_model(pnts_tensor)
            pred_fit = np.squeeze(pred_fit.detach().cpu().numpy())
            dsq_SE3, dsq_parameters = pred_to_parameters(pred_fit, normalizer, seg_info[1])
            dsq_SE3 = np.dot(pcd_frame, dsq_SE3)
            dsq_fit = DeformableSuperquadric(dsq_SE3, dsq_parameters)
            dsq_set.append(dsq_fit)

        # predicted obj
        result_sq = Object(sq_set, transform=False)
        result_dsq = Object(dsq_set, transform=False)

        # fitted objects coloring
        segment_number = np.max(pcd_seg) + 1
        random_clr_idx = np.random.choice(6, size=segment_number, replace=False)
        for p, primitive in enumerate(result_dsq.primitives):
            type_ = primitive.type
            primitive.mesh.paint_uniform_color(np.array(prim_clr_dict[random_clr_idx[p]]) / 255)
            if p == 0:
                result_mesh_dsqnet_colored = primitive.mesh
            else:
                result_mesh_dsqnet_colored = result_mesh_dsqnet_colored + primitive.mesh

        vertices_dsqnet = np.expand_dims(np.asarray(result_mesh_dsqnet_colored.vertices), axis=0)
        faces_dsqnet = np.expand_dims(np.asarray(result_mesh_dsqnet_colored.triangles), axis=0)
        colors_dsqnet = np.expand_dims(np.asarray(result_mesh_dsqnet_colored.vertex_colors), axis=0) * 255 
        writer.add_mesh(
            'dsqnet_results', 
            vertices = vertices_dsqnet, 
            faces = faces_dsqnet, 
            colors = colors_dsqnet, 
            global_step=1
        )

        # fitted objects coloring
        for p, primitive in enumerate(result_sq.primitives):
            type_ = primitive.type
            primitive.mesh.paint_uniform_color(np.array(prim_clr_dict[random_clr_idx[p]]) / 255)
            if p == 0:
                result_mesh_sqnet_colored = primitive.mesh
            else:
                result_mesh_sqnet_colored = result_mesh_sqnet_colored + primitive.mesh

        vertices_sqnet = np.expand_dims(np.asarray(result_mesh_sqnet_colored.vertices), axis=0)
        faces_sqnet = np.expand_dims(np.asarray(result_mesh_sqnet_colored.triangles), axis=0)
        colors_sqnet = np.expand_dims(np.asarray(result_mesh_sqnet_colored.vertex_colors), axis=0) * 255 
        writer.add_mesh(
            'sqnet_results', 
            vertices = vertices_sqnet, 
            faces = faces_sqnet, 
            colors = colors_sqnet, 
            global_step=1
        )

        # calculate iou
        if args.iou:
            iou_sqnet = iou(gt_mesh.mesh, result_sq.mesh)
            iou_dsqnet = iou(gt_mesh.mesh, result_dsq.mesh)
            print(f"[{file_name}] SQNet iou value: {iou_sqnet}")
            print(f"[{file_name}] DSQNet iou value: {iou_dsqnet}")

        # close writer
        writer.close()
