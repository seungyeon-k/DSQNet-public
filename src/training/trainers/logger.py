import numpy as np
import open3d as o3d
from metrics import averageMeter
# from training.functions.make_mesh import gt_mesh_from_primitives
# from training.functions.make_mesh import train_mesh_from_primitives
from training.functions.make_mesh import mesh_from_primitives, meshs_to_numpy, \
										 meshs_to_numpy_with_iou
from training.functions.lie import quaternions_to_rotation_matrices
# from training.functions.iou_calculator import iou

class BaseLogger:
	"""BaseLogger that can handle most of the logging
	logging convention
	------------------
	'loss' has to be exist in all training settings
	endswith('_') : scalar
	endswith('@') : image
	"""
	def __init__(self, tb_writer, **logger_cfg):
		"""tb_writer: tensorboard SummaryWriter"""
		self.writer = tb_writer
		self.logger_cfg = logger_cfg
		self.train_loss_meter = averageMeter()
		self.val_loss_meter = averageMeter()
		self.d_train = {}
		self.d_val = {}
		self.visualize_interval = logger_cfg['visualize_interval']
		self.visualize_number = logger_cfg['visualize_number']
		self.len_physical_params = logger_cfg['len_physical_params']
		self.primitive_type = logger_cfg['primitive_type']

	def process_iter_train(self, d_result):
		self.train_loss_meter.update(d_result['loss'])
		self.d_train = d_result

	def summary_train(self, i):
		self.d_train['loss/train_loss_'] = self.train_loss_meter.avg 
		for key, val in self.d_train.items():
			if key.endswith('_'):
				self.writer.add_scalar(key, val, i)
			if key.endswith('@') and i % self.visualize_interval == 0:
				if val is not None:
    					
					# make coordinate frame
					coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(
						size=0.2, origin=[0, 0, 0]
					)
					pc = val[0:self.visualize_number,:,:]
					pc_coor = np.asarray(coordinate.vertices)
					color_coor = np.asarray(coordinate.vertex_colors)
					pc_coor_repeat = np.tile(pc_coor, (pc.shape[0], 1, 1))
					color_coor_repeat = np.tile(color_coor, (pc.shape[0], 1, 1))
					pc_total = np.concatenate((pc, pc_coor_repeat), axis=1)
					color_total = np.concatenate((128 * np.ones(np.shape(pc)), 255 * color_coor_repeat), axis=1)
					self.writer.add_mesh(key, vertices=pc_total, colors = color_total, global_step=i)

			if key.endswith(')') and i % self.visualize_interval == 0:
				if val is not None:
    					
					# make coordinate frame
					coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(
						size=0.2, origin=[0, 0, 0]
					)
					pc = val[0:self.visualize_number,:,:]
					pc_coor = np.asarray(coordinate.vertices)
					color_coor = np.asarray(coordinate.vertex_colors)
					pc_coor_repeat = np.tile(pc_coor, (pc.shape[0], 1, 1))
					color_coor_repeat = np.tile(color_coor, (pc.shape[0], 1, 1))
					pc_total = np.concatenate((pc, pc_coor_repeat), axis=1)
					color_total = np.concatenate((128 * np.ones(np.shape(pc)), 255 * color_coor_repeat), axis=1)
					self.writer.add_mesh(key, vertices=pc_total, colors = color_total, global_step=i)
					# self.writer.add_mesh(key, vertices=val[0:self.visualize_number,:,:], global_step=i)

			if key.endswith('!') and i % self.visualize_interval == 0:
				if val is not None:

					# load
					pc = val[0]
					label = val[1]

					# color design
					label_total = np.concatenate((label==1, label==2, label==3), axis=2)
					color_wboundary = np.ones(np.shape(pc[0:self.visualize_number,:,:]))
					color_wboundary[label_total[0:self.visualize_number,:,:] == 1] = 0.1

					# make coordinate frame
					coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(
						size=0.2, origin=[0, 0, 0]
					)
					pc_coor = np.asarray(coordinate.vertices)
					color_coor = np.asarray(coordinate.vertex_colors)
					pc_coor_repeat = np.tile(pc_coor, (self.visualize_number, 1, 1))
					color_coor_repeat = np.tile(color_coor, (self.visualize_number, 1, 1))
					pc_total = np.concatenate((pc[0:self.visualize_number,:,:], pc_coor_repeat), axis=1)
					# color_total = np.concatenate((128 * np.ones(np.shape(pc[0:self.visualize_number,:,:])), 255 * color_coor_repeat), axis=1)
					color_total = np.concatenate((255 * color_wboundary, 255 * color_coor_repeat), axis=1)
					self.writer.add_mesh(key, vertices=pc_total, colors = color_total, global_step=i)
					# self.writer.add_mesh(key, vertices=pc[0:self.visualize_number,:,:], global_step=i)

			if key.endswith('$') and i % self.visualize_interval == 0:
				if val is not None:
					gt_vertices, gt_faces = mesh_from_primitives(val[0:self.visualize_number,:,:], 'groundtruth', **self.logger_cfg)     
					self.writer.add_mesh(key, vertices=gt_vertices, faces=gt_faces, global_step=i)
			if key.endswith('%') and i % self.visualize_interval == 0:
				if val is not None:
					train_vertices, train_faces = mesh_from_primitives(val[0:self.visualize_number,:,:], 'trained', **self.logger_cfg)
					self.writer.add_mesh(key, vertices=train_vertices, faces=train_faces, global_step=i)

			# for debugging
			if key.endswith('*') and i % self.visualize_interval == 0:
				if val is not None:
					gt_vertices, gt_faces = mesh_from_primitives(val[0][0:self.visualize_number,:,:], 'groundtruth', **self.logger_cfg)
					train_vertices, train_faces = mesh_from_primitives(val[1][0:self.visualize_number,:,:], 'trained', **self.logger_cfg)
					gt_colors = np.zeros(gt_vertices.shape)
					gt_colors[:,:,0] = 255
					train_colors = np.zeros(train_vertices.shape)
					train_colors[:,:,2] = 255

					self.writer.add_mesh(key, vertices=gt_vertices, faces=gt_faces, colors = gt_colors, global_step=i)
					self.writer.add_mesh(key, vertices=train_vertices, faces=train_faces, colors = train_colors, global_step=i)

					# for debugging
					batch_index = 0
					gt_meanx = np.mean(gt_vertices[batch_index,:,0])
					gt_meany = np.mean(gt_vertices[batch_index,:,1])
					gt_meanz = np.mean(gt_vertices[batch_index,:,2])
					gt_r = val[0][batch_index, 0, -3]
					gt_h = val[0][batch_index, 0, -2]
					train_meanx = np.mean(train_vertices[batch_index,:,0])
					train_meany = np.mean(train_vertices[batch_index,:,1])
					train_meanz = np.mean(train_vertices[batch_index,:,2])
					train_r = val[1][batch_index, 0, -2]
					train_h = val[1][batch_index, 0, -1]
					info_params = f"""# Results
					\ngt_meanx: {gt_meanx}
					\ngt_meany: {gt_meany}
					\ngt_meanz: {gt_meanz}
					\ngt_r: {gt_r}
					\ngt_h: {gt_h}
					\ntrain_meanx: {train_meanx}
					\ntrain_meany: {train_meany}
					\ntrain_meanz: {train_meanz}
					\ntrain_r: {train_r}
					\ntrain_h: {train_h}"""
					self.writer.add_text(key, info_params, i)

			# for debugging
			if key.endswith('~') and i % self.visualize_interval == 0:
				if val is not None:
					pc = val[0]
					y_gt = val[1]
					y_f = val[2]

					# difference
					y_diff = np.abs(y_gt[:, :, :3] - y_f[:, :, :3])
					y_max = np.max(y_diff)
					y_diff_color = y_diff / y_max * 254

					gt_position = pc[:self.visualize_number, 0, :3] + y_gt[:self.visualize_number, 0, :3]
					gt_orientation = y_gt[:self.visualize_number, 0, 3:6]
					gt_param = y_gt[:self.visualize_number, 0, 6:6+self.len_physical_params]
					confidence = y_f[:, :, 6+self.len_physical_params]
					estimate_position = pc[:self.visualize_number, :, :3] + y_f[:self.visualize_number, :, :3]
					estimate_orientation = y_f[:self.visualize_number, :, 3:6]
					estimate_param = y_f[:self.visualize_number, :, 6:6+self.len_physical_params]
					index = np.argsort(confidence, axis=1)
					confidence_max_index = index[:self.visualize_number, -1]
					# confidence_max_index = np.argmax(index[:self.visualize_number, :], axis=1)

					# truthful position and orientation
					truthful_position = []
					truthful_orientation = []
					truthful_param = []
					for batch in range(self.visualize_number):
						truthful_position.append(estimate_position[batch, confidence_max_index[batch], :])
						truthful_orientation.append(estimate_orientation[batch, confidence_max_index[batch], :])
						truthful_param.append(estimate_param[batch, confidence_max_index[batch], :])
					truthful_position = np.array(truthful_position)
					truthful_orientation = np.array(truthful_orientation)
					truthful_param = np.array(truthful_param)

					# calculate rotation matrix
					gt_R = []
					truthful_R = []
					for batch in range(self.visualize_number):
						gt_R.append(SO3_from_zaxis(gt_orientation[batch, :]))
						truthful_R.append(SO3_from_zaxis(truthful_orientation[batch, :]))
					gt_R = np.array(gt_R)
					truthful_R = np.array(truthful_R)

					# reconstruct mesh information
					mesh_info_gt = np.concatenate((np.ones((self.visualize_number,1)), gt_position, gt_R, gt_param), axis=1)
					mesh_info_train = np.concatenate((np.ones((self.visualize_number,1)), truthful_position, truthful_R, truthful_param), axis=1)
					mesh_info_gt = np.expand_dims(mesh_info_gt, axis=1)
					mesh_info_train = np.expand_dims(mesh_info_train, axis=1)

					# reconstruct mesh
					gt_mesh = mesh_from_primitives(
						mesh_info_gt[0:self.visualize_number,:,:], 
						split='trained',
						dtype='o3d', # dtype='numpy',
						**self.logger_cfg
					)
					train_mesh = mesh_from_primitives(
						mesh_info_train[0:self.visualize_number,:,:], 
						split='trained', 
						dtype='o3d', # dtype='numpy',
						**self.logger_cfg
					)

					# gt_vertices, gt_faces = mesh_from_primitives(
					# 	mesh_info_gt[0:self.visualize_number,:,:], 
					# 	split='trained',
					# 	dtype='numpy',
					# 	**self.logger_cfg
					# )
					# train_vertices, train_faces = mesh_from_primitives(
					# 	mesh_info_train[0:self.visualize_number,:,:], 
					# 	split='trained', 
					# 	dtype='numpy',
					# 	**self.logger_cfg
					# )
					# total_vertices = np.concatenate((gt_vertices, train_vertices), axis=1)
					# total_faces = np.concatenate((gt_faces, train_faces), axis=1)

					total_vertices, total_faces, total_colors = meshs_to_numpy(gt_mesh, train_mesh)
					
					# # color information
					# gt_colors = np.zeros(gt_vertices.shape)
					# gt_colors[:,:,0] = 255
					# train_colors = np.zeros(train_vertices.shape)
					# train_colors[:,:,2] = 255
					# total_colors = np.concatenate((gt_colors, train_colors), axis=1)

					# draw mesh
					self.writer.add_mesh(
						key, 
						vertices=total_vertices, 
						faces=total_faces, 
						colors = total_colors, 
						global_step=i
					)

					# text
					for j in range(self.visualize_number):
						for k in range(1, 2):
							info_params = f"""# sample number {j+1} 
							\n ## confidence order {k}
							\n gt_position: {gt_position[j, :3]}
							\n gt_orientation: {gt_orientation[j, :3]}
							\n confidence_lelvel: {confidence[j, confidence_max_index[j]]}
							\n estimate_position: {estimate_position[j, confidence_max_index[j], :3]}
							\n estimate_orientation: {estimate_orientation[j, confidence_max_index[j], :3]}
							"""
							self.writer.add_text(key, info_params, i)

			# for debugging
			if key.endswith('#') and i % self.visualize_interval == 0:
				if val is not None:
					pc = val[0]
					y_gt = val[1]
					y_f = val[2]

					estimate_position = y_f[:self.visualize_number, :3]
					estimate_quaternion = y_f[:self.visualize_number, 3:7]
					estimate_orientation = quaternions_to_rotation_matrices(estimate_quaternion).reshape(-1, 9)
					estimate_param = y_f[:self.visualize_number, 7:]

					# reconstruct mesh information
					n_visualize_number = estimate_param.shape[0]
					mesh_info_train = np.concatenate((np.ones((n_visualize_number,1)), estimate_position, estimate_orientation, estimate_param), axis=1)
					# mesh_info_train = np.concatenate((np.ones((self.visualize_number,1)), estimate_position, estimate_orientation, estimate_param), axis=1)
					mesh_info_train = np.expand_dims(mesh_info_train, axis=1)
					mesh_info_gt = y_gt

					# reconstruct mesh
					gt_mesh = mesh_from_primitives(
						mesh_info_gt[0:self.visualize_number,:,:], 
						split='groundtruth',
						dtype='o3d', # dtype='numpy',
						info_types_from='cfg',
						**self.logger_cfg
					)
					train_mesh = mesh_from_primitives(
						mesh_info_train[0:self.visualize_number,:,:], 
						split='trained', 
						dtype='o3d', # dtype='numpy',
						info_types_from=self.primitive_type,
						**self.logger_cfg
					)

					total_vertices, total_faces, total_colors = meshs_to_numpy(gt_mesh, train_mesh, bcoordinate = True)

					# draw mesh
					self.writer.add_mesh(
						key, 
						vertices=total_vertices, 
						faces=total_faces, 
						colors = total_colors, 
						global_step=i
					)

					# text
					if self.primitive_type == 'superquadric':		
						info_params = f"""# sample number {1} 
						\n a1: {estimate_param[0, 0]}
						\n a2: {estimate_param[0, 1]}
						\n a3: {estimate_param[0, 2]}
						\n e1: {estimate_param[0, 3]}
						\n e2: {estimate_param[0, 4]}
						\n x: {estimate_position[0, 0]}
						\n y: {estimate_position[0, 1]}
						\n z: {estimate_position[0, 2]}
						\n x_gt: {y_gt[0, 0, 1]}
						\n y_gt: {y_gt[0, 0, 2]}
						\n z_gt: {y_gt[0, 0, 3]}
						"""

					elif self.primitive_type == 'extended_superquadric':
						info_params = f"""# sample number {1} 
						\n a1: {estimate_param[0, 0]}
						\n a2: {estimate_param[0, 1]}
						\n a3: {estimate_param[0, 2]}
						\n e1: {estimate_param[0, 3]}
						\n e2: {estimate_param[0, 4]}
						\n c1: {estimate_param[0, 5]}
						\n c2: {estimate_param[0, 6]}
						\n x: {estimate_position[0, 0]}
						\n y: {estimate_position[0, 1]}
						\n z: {estimate_position[0, 2]}
						\n R: {estimate_orientation[0, :]}
						"""

					elif self.primitive_type == 'supertoroid':
						info_params = f"""# sample number {1} 
						\n a1: {estimate_param[0, 0]}
						\n a2: {estimate_param[0, 1]}
						\n a3: {estimate_param[0, 2]}
						\n a4: {estimate_param[0, 3]}
						\n e1: {estimate_param[0, 4]}
						\n e2: {estimate_param[0, 5]}
						\n x: {estimate_position[0, 0]}
						\n y: {estimate_position[0, 1]}
						\n z: {estimate_position[0, 2]}
						\n R: {estimate_orientation[0, :]}
						"""

					elif self.primitive_type == 'extended_supertoroid':
    						info_params = f"""# sample number {1} 
						\n a1: {estimate_param[0, 0]}
						\n a2: {estimate_param[0, 1]}
						\n a3: {estimate_param[0, 2]}
						\n a4: {estimate_param[0, 3]}
						\n e1: {estimate_param[0, 4]}
						\n e2: {estimate_param[0, 5]}
						\n c1: {estimate_param[0, 6]}
						\n x: {estimate_position[0, 0]}
						\n y: {estimate_position[0, 1]}
						\n z: {estimate_position[0, 2]}
						\n R: {estimate_orientation[0, :]}
						"""

					elif self.primitive_type == 'deformed_superquadric':
    						info_params = f"""# sample number {1} 
						\n a1: {estimate_param[0, 0]}
						\n a2: {estimate_param[0, 1]}
						\n a3: {estimate_param[0, 2]}
						\n e1: {estimate_param[0, 3]}
						\n e2: {estimate_param[0, 4]}
						\n k: {estimate_param[0, 5]}
						\n b: {estimate_param[0, 6]}
						\n x: {estimate_position[0, 0]}
						\n y: {estimate_position[0, 1]}
						\n z: {estimate_position[0, 2]}
						\n R: {estimate_orientation[0, :]}
						"""

					self.writer.add_text(key, info_params, i)

		result = self.d_train
		self.d_train = {}
		return result

	def process_iter_val(self, d_result):
		self.val_loss_meter.update(d_result['loss'])
		self.d_val = d_result

	def summary_val(self, i):
		self.d_val['loss/val_loss_'] = self.val_loss_meter.avg 
		l_print_str = [f'Iter [{i:d}]']
		for key, val in self.d_val.items():
			if key.endswith('_'):
				self.writer.add_scalar(key, val, i)
				l_print_str.append(f'{key}: {val:.4f}')
			if key.endswith('@'):
				if val is not None:
					self.writer.add_mesh(key, vertices=val[0:self.visualize_number,:,:], global_step=i)
			if key.endswith(')') and i % self.visualize_interval == 0:
				if val is not None:
					self.writer.add_mesh(key, vertices=val[0:self.visualize_number,:,:], global_step=i)
			# for debugging
			if key.endswith('~') and i % self.visualize_interval == 0:
				if val is not None:
					pc = val[0]
					y_gt = val[1]
					y_f = val[2]

					# difference
					y_diff = np.abs(y_gt[:, :, :3] - y_f[:, :, :3])
					y_max = np.max(y_diff)
					y_diff_color = y_diff / y_max * 254

					gt_position = pc[:self.visualize_number, 0, :3] + y_gt[:self.visualize_number, 0, :3]
					gt_orientation = y_gt[:self.visualize_number, 0, 3:6]
					gt_param = y_gt[:self.visualize_number, 0, 6:6+self.len_physical_params]
					confidence = y_f[:, :, 6+self.len_physical_params]
					estimate_position = pc[:self.visualize_number, :, :3] + y_f[:self.visualize_number, :, :3]
					estimate_orientation = y_f[:self.visualize_number, :, 3:6]
					estimate_param = y_f[:self.visualize_number, :, 6:6+self.len_physical_params]
					index = np.argsort(confidence, axis=1)
					confidence_max_index = index[:self.visualize_number, -1]

					# truthful position and orientation
					truthful_position = []
					truthful_orientation = []
					truthful_param = []
					for batch in range(self.visualize_number):
						truthful_position.append(estimate_position[batch, confidence_max_index[batch], :])
						truthful_orientation.append(estimate_orientation[batch, confidence_max_index[batch], :])
						truthful_param.append(estimate_param[batch, confidence_max_index[batch], :])
					truthful_position = np.array(truthful_position)
					truthful_orientation = np.array(truthful_orientation)
					truthful_param = np.array(truthful_param)

					# calculate rotation matrix
					gt_R = []
					truthful_R = []
					for batch in range(self.visualize_number):
						gt_R.append(SO3_from_zaxis(gt_orientation[batch, :]))
						truthful_R.append(SO3_from_zaxis(truthful_orientation[batch, :]))
					gt_R = np.array(gt_R)
					truthful_R = np.array(truthful_R)

					# reconstruct mesh information
					mesh_info_gt = np.concatenate((np.ones((self.visualize_number,1)), gt_position, gt_R, gt_param), axis=1)
					mesh_info_train = np.concatenate((np.ones((self.visualize_number,1)), truthful_position, truthful_R, truthful_param), axis=1)
					mesh_info_gt = np.expand_dims(mesh_info_gt, axis=1)
					mesh_info_train = np.expand_dims(mesh_info_train, axis=1)

					# reconstruct mesh
					gt_mesh = mesh_from_primitives(
						mesh_info_gt[0:self.visualize_number,:,:], 
						split='trained',
						dtype='o3d', # dtype='numpy',
						**self.logger_cfg
					)
					train_mesh = mesh_from_primitives(
						mesh_info_train[0:self.visualize_number,:,:], 
						split='trained', 
						dtype='o3d', # dtype='numpy',
						**self.logger_cfg
					)

					# gt_vertices, gt_faces = mesh_from_primitives(
					# 	mesh_info_gt[0:self.visualize_number,:,:], 
					# 	split='trained',
					# 	dtype='numpy',
					# 	**self.logger_cfg
					# )
					# train_vertices, train_faces = mesh_from_primitives(
					# 	mesh_info_train[0:self.visualize_number,:,:], 
					# 	split='trained', 
					# 	dtype='numpy',
					# 	**self.logger_cfg
					# )
					# total_vertices = np.concatenate((gt_vertices, train_vertices), axis=1)
					# total_faces = np.concatenate((gt_faces, train_faces), axis=1)

					total_vertices, total_faces, total_colors = meshs_to_numpy(gt_mesh, train_mesh)
					
					# # color information
					# gt_colors = np.zeros(gt_vertices.shape)
					# gt_colors[:,:,0] = 255
					# train_colors = np.zeros(train_vertices.shape)
					# train_colors[:,:,2] = 255
					# total_colors = np.concatenate((gt_colors, train_colors), axis=1)

					# draw mesh
					self.writer.add_mesh(
						key, 
						vertices=total_vertices, 
						faces=total_faces, 
						colors = total_colors, 
						global_step=i
					)

			# for debugging
			if key.endswith('#') and i % self.visualize_interval == 0:
				if val is not None:
					pc = val[0]
					y_gt = val[1]
					y_f = val[2]

					estimate_position = y_f[:self.visualize_number, :3]
					estimate_quaternion = y_f[:self.visualize_number, 3:7]
					estimate_orientation = quaternions_to_rotation_matrices(estimate_quaternion).reshape(-1, 9)
					estimate_param = y_f[:self.visualize_number, 7:]

					# reconstruct mesh information
					n_visualize_number = estimate_param.shape[0]
					mesh_info_train = np.concatenate((np.ones((n_visualize_number,1)), estimate_position, estimate_orientation, estimate_param), axis=1)
					mesh_info_train = np.expand_dims(mesh_info_train, axis=1)
					mesh_info_gt = y_gt

					# reconstruct mesh
					gt_mesh = mesh_from_primitives(
						mesh_info_gt[0:self.visualize_number,:,:], 
						split='trained',
						dtype='o3d', # dtype='numpy',
						info_types_from='cfg',
						**self.logger_cfg
					)
					train_mesh = mesh_from_primitives(
						mesh_info_train[0:self.visualize_number,:,:], 
						split='trained', 
						dtype='o3d', # dtype='numpy',
						info_types_from=self.primitive_type,
						**self.logger_cfg
					)

					# # calculate iou
					# iou_batch = []
					# for batch in range(n_visualize_number):
					# 	iou_ = iou(gt_mesh[batch], train_mesh[batch], voxel_resolution=40)
					# 	iou_batch.append(iou_)
					# 	print(f"{batch + 1}th batch's iou calculation is completed")

					total_vertices, total_faces, total_colors = meshs_to_numpy(gt_mesh, train_mesh, bcoordinate = True)
					# total_vertices, total_faces, total_colors = meshs_to_numpy_with_iou(gt_mesh, train_mesh, iou_batch)

					# draw mesh
					self.writer.add_mesh(
						key, 
						vertices=total_vertices, 
						faces=total_faces, 
						colors = total_colors, 
						global_step=i
					)

					# # add iou metric
					# self.writer.add_scalar('iou_val', sum(iou_batch) / len(iou_batch), i)
			

		print_str = ' '.join(l_print_str)

		result = self.d_val
		result['print_str'] = print_str
		self.d_val = {}
		return result

def SO3_from_zaxis(z):

	x = np.random.randn(3)  # take a random vector
	x -= x.dot(z) * z       # make it orthogonal to k
	x /= np.linalg.norm(x)  # normalize it
	y = np.cross(z, x)
	R = np.array([x, y, z]).transpose()
	R_vec = R[0,:].tolist() + R[1,:].tolist() + R[2,:].tolist()
	R_vec = np.array(R_vec)

	return R_vec
