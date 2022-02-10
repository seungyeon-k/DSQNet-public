import open3d as o3d
import numpy as np
import sys
import training.functions.lie_alg as lie_alg
import copy
# sys.path.append('./data_generation/functions/')
# from util import transform_to_pc_frame
# import pclpy
# import pcl
# import pcl
import pclpy
from sklearn.preprocessing import normalize

def voxel_downsample_npoints(pcd, num_pcd_points):
	v_size = 0.5
	v_min = 0
	v_max = 1
	v_last_met_requirements = None
	cnt = 0
	for trial in range(100):
		num_tmp_pcd_pnts = np.asarray(pcd.voxel_down_sample(v_size).points).shape[0]
		if num_tmp_pcd_pnts - num_pcd_points >= 0 and num_tmp_pcd_pnts - num_pcd_points < 10:
			break 
		if num_tmp_pcd_pnts > num_pcd_points:
			v_last_met_requirements = v_size
			v_min = v_size

		if num_tmp_pcd_pnts < num_pcd_points:
			v_max = v_size
		v_size = (v_min + v_max) / 2
		if trial == 99 and num_tmp_pcd_pnts > num_pcd_points:
			if v_last_met_requirements is not None:
				v_size = v_last_met_requirements
			else:
				v_size = v_min
		cnt += 1

	pcd = pcd.voxel_down_sample(v_size)
	pcd = pcd.select_by_index([*range(num_pcd_points)])

	return pcd

def upsample_pc(pcd):
	# pcd_tree = o3d.geometry.KDTreeFlann(pcd)
	# nn_pnts_list = []
	# for pnt_ind in range(len(pcd.points)):
	# 	a, nn_pnt_ind, c = pcd_tree.search_knn_vector_3d(pcd.points[pnt_ind], 3)
	# 	nn_pnts_list.append(list(np.sort(list(np.asarray(nn_pnt_ind)))))

	# nn_pnts_set = set(map(tuple, nn_pnts_list))  #need to convert the inner lists to tuples so they are hashable
	# nn_pnts_list = list(map(list, nn_pnts_set))
	# nn_pnts_np = np.array(nn_pnts_list)
	# pnts = np.asarray(pcd.points)
	# interpolated_pnts = (pnts[nn_pnts_np[:, 0], :] + pnts[nn_pnts_np[:, 1], :] + pnts[nn_pnts_np[:, 2], :]) / 3
	# pnts = np.concatenate((pnts, interpolated_pnts), axis=0)
	# pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pnts))
	# return pcd

	# point_cloud = pcl.PointCloud(np.asarray(pcd.points, dtype=np.float32))
	# tree = point_cloud.make_kdtree()
	# mls = point_cloud.make_moving_least_squares()
	# mls.set_Compute_Normals(True)
	# mls.set_polynomial_fit(True)
	# mls.set_Search_Method(tree)
	# mls.set_search_radius(0.01)
	# mls.set_upsampling_radius(0.002)
	# mls.setUpsamplingStepSize(0.001)
	# mls_points = mls.process()
	
	# pcd.points = o3d.utility.Vector3dVector(np.asarray(mls_points))
	
	
	# point_cloud = pclpy.pcl.PointCloud.PointXYZ(np.asarray(pcd.points))
	# cloud_type = pclpy.utils.get_point_cloud_type(point_cloud)
	
	# # output_cloud = getattr(pclpy.pcl.PointCloud, cloud_type)()
	# output_cloud = pclpy.pcl.PointCloud.PointXYZ()
	# print(1)
	# mls = pclpy.pcl.surface.MovingLeastSquares.PointXYZ_PointXYZ()
	# print(2)
	# mls.setInputCloud(point_cloud)
	# print(3)
	# mls.setComputeNormals(True)
	# print(4)
	# mls.setPolynomialOrder(2)
	# print(5)
	# mls.setSearchRadius(0.01)
	# print(6)
	# mls.setUpsamplingMethod(pclpy.pcl.surface.MovingLeastSquares.PointXYZ_PointXYZ.UpsamplingMethod.SAMPLE_LOCAL_PLANE)
	# print(7)
	# mls.setUpsamplingRadius(0.002)
	# print(8)
	# mls.setUpsamplingStepSize(0.001)
	# print(9)
	# mls.process(output_cloud)
	# # output_cloud = mls.process()
	# print(10)

	# pcd.points = o3d.utility.Vector3dVector(output_cloud.xyz)

	normalized_points, normalizer = normalize_numpy_pc(np.asarray(pcd.points))
	pcd.points = o3d.utility.Vector3dVector(normalized_points)
	# pcd = voxel_downsample_npoints(pcd, int(np.floor(0.4 * len(pcd.points))))
	pcd.estimate_normals()
	# radii = [0.0005, 0.001 , 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02]
	#radii = [0.001, 0.003, 0.006, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
	radii = np.arange(0.0001, 0.1, 0.0001).tolist()
	surf = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
	o3d.visualization.draw_geometries([surf, pcd])
	sampled_pcd = surf.sample_points_uniformly(1500)
	pcd.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd.points), np.asarray(sampled_pcd.points)),axis=0) * normalizer)

	# normalized_points, normalizer = normalize_numpy_pc(np.asarray(pcd.points))
	# pcd.points = o3d.utility.Vector3dVector(normalized_points)

	# pcd.estimate_normals()
	# pcd_numpy = np.asarray(pcd.points)
	# pcd_normals_numpy = np.asarray(pcd.normals)
	# random_idx = np.random.choice(len(pcd_numpy), size=3000, replace = True)
	# pcd_numpy = pcd_numpy[random_idx].T
	# pcd_numpy = pcd_numpy.T
	# pcd_normals_numpy = pcd_normals_numpy[random_idx].T
	# pcd_normals_numpy = pcd_normals_numpy.T

	# noise = np.random.uniform(-1, 1, size=pcd_numpy.shape)
	# noise = normalize(noise, axis=0, norm='l2')
	# print(pcd_normals_numpy.shape, noise.shape)
	# noise = np.cross(pcd_normals_numpy, noise)
	# print(noise.shape)
	# noise = noise / np.linalg.norm(noise)
	# noise_std = 0.1
	# scale = np.random.normal(loc=0, scale=noise_std, size=(1, pcd_numpy.shape[1])).repeat(pcd_numpy.shape[0], axis=0)
	# pcd_numpy = pcd_numpy + noise * scale
	# pcd.points = o3d.utility.Vector3dVector(pcd_numpy)
	# # pcd.points = o3d.utility.Vector3dVector(pcd_numpy * normalizer)
	# pcd.estimate_normals()
	# o3d.visualization.draw_geometries([pcd])
	
	return pcd

def process_pc(pcd, num_processed_pc_points=3000):
	num_pnts = len(pcd.points)
	
	# upsample
	if num_pnts < num_processed_pc_points:
		pcd = upsample_pc(pcd)
	
	# downsample
	try:
		pcd = voxel_downsample_npoints(pcd, num_processed_pc_points)
	except:
		pcd = upsample_pc(pcd)
		pcd = voxel_downsample_npoints(pcd, num_processed_pc_points)
	
	# transform to pc frame
	pcd, pcd_frame = transform_to_pc_frame(pcd)

	return pcd, pcd_frame

def normalize_numpy_pc(pnts):
	max_ = np.max(pnts, axis=0)
	min_ = np.min(pnts, axis=0)
	diagonal_len = np.linalg.norm(max_-min_)
	pnts = pnts / diagonal_len
	normalizer = diagonal_len

	return pnts, normalizer

if __name__ == '__main__':

	file = '20201119-222638_seg'
	
	# load pc
	pnts = np.load(f"./grasping/real_data/{file}.npy")
	print("number of points before process: ", len(pnts))
	pnts = pnts[:2800, 0:3]
	pcd = o3d.geometry.PointCloud(points = o3d.utility.Vector3dVector(pnts))

	# process pc
	pcd = process_pc(pcd, num_processed_pc_points=3000)
	print("number of points after process: ", len(pcd.points))

	# draw pc
	origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01)
	o3d.visualization.draw_geometries([pcd, origin_frame])
	
	# save processed pc
	#np.save(f"./grasping/real_data/processed/{obj}_processed.npy", np.asarray(pcd.points))

def transform_to_pc_frame(pcd):
  
    try:
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
    except:
        # print('----------2d convexhull error----------')
        # pnts = np.asarray(pcd.points)
        # pnts[0, 2] += 0.001
        # pcd.points = o3d.utility.Vector3dVector(pnts)
        # bbox = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
        return None, None

    if np.abs(np.linalg.det(bbox.R) + 1) < 1e-4:
        object_frame = lie_alg.define_SE3(-bbox.R, bbox.center)
    else:
        object_frame = lie_alg.define_SE3(bbox.R, bbox.center)

    T_xyz_to_zxy = lie_alg.define_SE3(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), [0, 0, 0])
    object_frame = np.matmul(object_frame, T_xyz_to_zxy)
    pcd_cand = copy.deepcopy(pcd)
    pcd_cand.transform(lie_alg.inverse_SE3(object_frame))
    
    # flip frame so that pnts below xy plane has larger volume
    pnts = np.asarray(pcd_cand.points)
    proj_xy_norms = np.linalg.norm(pnts[:, 0:2], axis=1)
    proj_xy_norms_upper_pnts = proj_xy_norms[np.where(pnts[:, 2] > 0)]
    proj_xy_norms_lower_pnts = proj_xy_norms[np.where(pnts[:, 2] < 0)]
    avg_xy_norm_upper = np.mean(proj_xy_norms_upper_pnts)
    avg_xy_norm_lower = np.mean(proj_xy_norms_lower_pnts)
    if avg_xy_norm_upper > avg_xy_norm_lower:
        T_flip_x = lie_alg.define_SE3(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), [0, 0, 0])
        object_frame = np.matmul(object_frame, T_flip_x)
        pcd_cand = copy.deepcopy(pcd)
        pcd_cand.transform(lie_alg.inverse_SE3(object_frame))

    pcd_cand2 = copy.deepcopy(pcd_cand)
    pnts = np.asarray(pcd_cand2.points)
    proj_yz_norms = np.linalg.norm(pnts[:, 1:3], axis=1)
    proj_yz_norms_upper_pnts = proj_yz_norms[np.where(pnts[:, 0] > 0)]
    proj_yz_norms_lower_pnts = proj_yz_norms[np.where(pnts[:, 0] < 0)]
    avg_yz_norm_upper = np.mean(proj_yz_norms_upper_pnts)
    avg_yz_norm_lower = np.mean(proj_yz_norms_lower_pnts)
    if avg_yz_norm_upper > avg_yz_norm_lower:
        T_flip_z = lie_alg.define_SE3(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), [0, 0, 0])
        object_frame = np.matmul(object_frame, T_flip_z)
        pcd_cand2 = copy.deepcopy(pcd)
        pcd_cand2.transform(lie_alg.inverse_SE3(object_frame))
    
    pcd = copy.deepcopy(pcd_cand2)
    return pcd, object_frame