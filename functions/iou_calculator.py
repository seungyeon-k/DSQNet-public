import numpy as np
from numpy.linalg import inv
from copy import deepcopy
import open3d as o3d
# import functions.primitives as primitives
# from functions.object_class import Object
# from training.functions.lie import exp_se3 
# from training.functions.primitives import Box, Cone, Cylinder, Sphere, Torus, RectangleRing, CylinderRing, \
#                                           Superquadric, Extended_Superquadric, \
#                                           Supertoroid, Extended_Supertoroid, \
#                                           Semi_Sphere_Shell, \
#                                           Deformed_Superquadric
# import trimesh
# import pymesh

def get_voxel_indices(voxel):
    indices = []
    for v in voxel.get_voxels():
        indices.append(v.grid_index)
    indices = np.array(indices)
    # print(voxel)
    # print(indices)
    return indices

def voxelization_with_inner_volume(mesh, voxel_size, grid_bound):
    voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size, grid_bound[0], grid_bound[1])

    indices = get_voxel_indices(voxel)

    indices_with_inside = []
    for z in range(min(indices[:, 2]), max(indices[:, 2])):
        indices_with_z = indices[np.squeeze(np.argwhere(indices[:, 2] == z))]
        if len(indices_with_z.shape) == 1:
            indices_with_inside.append(indices_with_z.tolist())
        else:
            for y in range(min(indices_with_z[:, 1]), max(indices_with_z[:, 1])):
                indices_with_y = indices_with_z[np.squeeze(np.argwhere(indices_with_z[:, 1] == y))]
                if len(indices_with_y.shape) > 1:
                    indices_to_add = np.concatenate((np.reshape(np.arange(min(indices_with_y[:, 0]), max(indices_with_y[:, 0]) + 1, dtype=int), (-1, 1)), y * np.ones((max(indices_with_y[:, 0]) - min(indices_with_y[:, 0]) + 1, 1), dtype=int), z * np.ones((max(indices_with_y[:, 0]) - min(indices_with_y[:, 0]) + 1, 1), dtype=int)), axis=1)
                    indices_with_inside = indices_with_inside + indices_to_add.tolist()
                else:
                    indices_with_inside.append(indices_with_y.tolist())

    indices_with_inside = np.array(indices_with_inside)
    pnts = np.tile(voxel.origin, (indices_with_inside.shape[0], 1)) + voxel_size * indices_with_inside
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(np.array(pnts)))

    voxel = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd, voxel_size, grid_bound[0], grid_bound[1])
    return voxel

def voxel_iou(voxel1, voxel2):
    indices1 = get_voxel_indices(voxel1)
    indices2 = get_voxel_indices(voxel2)
    
    dim = np.max(np.array([np.max(indices1, axis=0), np.max(indices2, axis=0)]), axis=0)

    voxel_bool1 = np.zeros(dim + 1)
    voxel_bool2 = np.zeros(dim + 1)

    try:
        voxel_bool1[indices1[:,0], indices1[:,1], indices1[:,2]] = 1
    except:
        print('voxel1 boolean error!!')

    try:
        voxel_bool2[indices2[:,0], indices2[:,1], indices2[:,2]] = 1
    except:
        print('voxel2 boolean error!!')
    
    intersection = np.sum(np.logical_and(voxel_bool1, voxel_bool2))
    union = np.sum(np.logical_or(voxel_bool1, voxel_bool2))
        
    return float(intersection) / float(union)

def iou(mesh1, mesh2, voxel_resolution=40, camera_resolution=3):
    
    # calculate scale factor 
    bound_min = np.min(np.array([mesh1.get_min_bound(), mesh2.get_min_bound()]), axis=0)
    bound_max = np.max(np.array([mesh1.get_max_bound(), mesh2.get_max_bound()]), axis=0)
    scale = np.max(bound_max - bound_min)
    center = bound_min + (bound_max - bound_min) / 2

    # mesh scaling
    mesh1_preprocessed = preprocess(mesh1, scale, center)
    mesh2_preprocessed = preprocess(mesh2, scale, center)

    # recalculate bound
    bound_min = np.min(np.array([mesh1_preprocessed.get_min_bound(), mesh2_preprocessed.get_min_bound()]), axis=0)
    bound_max = np.max(np.array([mesh1_preprocessed.get_max_bound(), mesh2_preprocessed.get_max_bound()]), axis=0)
    voxel_size = np.min(bound_max - bound_min) / voxel_resolution
    bound_min = bound_min - 2 * voxel_size
    bound_max = bound_max + 2 * voxel_size
    grid_bound = np.array([bound_min, bound_max]).transpose()

    voxel1, _, _ = voxel_carving(mesh1_preprocessed, grid_bound, voxel_resolution + 4, camera_resolution)
    voxel2, _, _ = voxel_carving(mesh2_preprocessed, grid_bound, voxel_resolution + 4, camera_resolution)
    return voxel_iou(voxel1, voxel2)

# def iou_trimesh(mesh1, mesh2):

#     return intersection_over_union(mesh1, mesh2)[0]

# def intersection_over_union(mesh1, mesh2):
#     union_ = trimesh.boolean.union((mesh1, mesh2), engine='blender')
#     intersection_ = trimesh.boolean.intersection((mesh1, mesh2), engine='blender')
#     # print(intersection.volume / union.volume,intersection.volume, union.volume)
#     # union.show()
#     # intersection.show()
#     return intersection_.volume / union_.volume, intersection_, union_

# def iou_pymesh(mesh1, mesh2):

#         mesh1_pymesh = pymesh.form_mesh(np.asarray(mesh1.vertices), np.asarray(mesh1.triangles))
#         grid1 = pymesh.VoxelGrid(2./dim)
#         grid1.insert_mesh(mesh1_pymesh)
#         grid1.create_grid()

#         ind1 = ((grid1.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
#         v1 = np.zeros([dim, dim, dim])
#         v1[ind1[:,0], ind1[:,1], ind1[:,2]] = 1

#         mesh2_pymesh = pymesh.form_mesh(np.asarray(mesh2.vertices), np.asarray(mesh2_pymesh.triangles))
#         grid2 = pymesh.VoxelGrid(2./dim)
#         grid2.insert_mesh(mesh2_pymesh)
#         grid2.create_grid()

#         ind2 = ((grid2.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
#         v2 = np.zeros([dim, dim, dim])
#         v2[ind2[:,0], ind2[:,1], ind2[:,2]] = 1

#         intersection = np.sum(np.logical_and(v1, v2))
#         union = np.sum(np.logical_or(v1, v2))

#         return float(intersection) / union

def xyz_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    r_x = np.arccos(y / r)
    r_y = np.arctan2(z, x)
    return [r, r_x, r_y]


def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                        [0, np.sin(r_x), np.cos(r_x)]])
    rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0],
                        [-np.sin(r_y), 0, np.cos(r_y)]])
    return rot_y.dot(rot_x)

def get_extrinsic(xyz):
    rvec = xyz_spherical(xyz)
    r = get_rotation_matrix(rvec[1], rvec[2])
    t = np.asarray([0, 0, 2]).transpose()
    trans = np.eye(4)
    trans[:3, :3] = r
    trans[:3, 3] = t
    return trans

def preprocess(model, scale, center):
    # min_bound = model.get_min_bound()
    # max_bound = model.get_max_bound()
    # center = min_bound + (max_bound - min_bound) / 2.0
    # scale = np.linalg.norm(max_bound - min_bound) / 2.0
    model_ = deepcopy(model)
    vertices = np.asarray(model_.vertices)
    vertices -= center
    model_.vertices = o3d.utility.Vector3dVector(vertices / scale)
    return model_

def voxel_carving(mesh,
                  grid_bound,
                  voxel_resolution = 0,
                  camera_resolution = 3,
                  w=300,
                  h=300,
                  use_depth=True,
                  surface_method='pointcloud'):
    mesh.compute_vertex_normals()
    cubic_size = grid_bound[:, 1] - grid_bound[:, 0]
    camera_sphere = o3d.geometry.TriangleMesh.create_sphere(1, resolution=camera_resolution)

    # setup dense voxel grid
    voxel_carving = o3d.geometry.VoxelGrid.create_dense(
        width=cubic_size[0],
        height=cubic_size[1],
        depth=cubic_size[2],
        voxel_size=np.min(cubic_size) / voxel_resolution,
        origin=grid_bound[:, 0].tolist(),
        color=[1.0, 0.7, 0.0]
        )

    # rescale geometry
    # camera_sphere = preprocess(camera_sphere)
    # mesh = preprocess(mesh)

    # setup visualizer to render depthmaps
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    # carve voxel grid
    pcd_agg = o3d.geometry.PointCloud()
    centers_pts = np.zeros((len(camera_sphere.vertices), 3))
    for cid, xyz in enumerate(camera_sphere.vertices):
        # get new camera pose
        trans = get_extrinsic(xyz)
        param.extrinsic = trans
        c = np.linalg.inv(trans).dot(np.asarray([0, 0, 0, 1]).transpose())
        centers_pts[cid, :] = c[:3]
        ctr.convert_from_pinhole_camera_parameters(param)

        # capture depth image and make a point cloud
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(False)
        pcd_agg += o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth),
            param.intrinsic,
            param.extrinsic,
            depth_scale=1)

        # depth map carving method
        if use_depth:
            voxel_carving.carve_depth_map(o3d.geometry.Image(depth), param)
        else:
            voxel_carving.carve_silhouette(o3d.geometry.Image(depth), param)
        # print("Carve view %03d/%03d" % (cid + 1, len(camera_sphere.vertices)))
    vis.destroy_window()

    # add voxel grid survace
    # print('Surface voxel grid from %s' % surface_method)
    if surface_method == 'pointcloud':
        voxel_surface = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd_agg,
            voxel_size=np.min(cubic_size) / voxel_resolution,
            min_bound=tuple(grid_bound[:, 0]),
            max_bound=tuple(grid_bound[:, 1]))
    elif surface_method == 'mesh':
        voxel_surface = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=tuple(grid_bound[:, 0]),
            max_bound=tuple(grid_bound[:, 1]))
    else:
        raise Exception('invalid surface method')
    voxel_carving_surface = voxel_surface + voxel_carving

    return voxel_carving_surface, voxel_carving, voxel_surface

if __name__ == '__main__':
    sphere = o3d.geometry.TriangleMesh.create_torus(0.5, 0.1)
    sphere2 = o3d.geometry.TriangleMesh.create_torus(0.5, 0.05)
    # voxel_grid, voxel_carving, voxel_surface = voxel_carving(sphere, cubic_size = 2.0, voxel_resolution = 128)
    # print(voxel_grid)
    # print(voxel_carving)
    # print(voxel_surface)