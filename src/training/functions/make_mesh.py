import numpy as np
from numpy.linalg import inv
from copy import deepcopy
import open3d as o3d
# import functions.primitives as primitives
# from functions.object_class import Object
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion
from training.functions.lie import exp_se3 
from training.functions.primitives import Box, Cone, Cylinder, Sphere, Torus, RectangleRing, CylinderRing, \
                                          Superquadric, Extended_Superquadric, \
                                          Supertoroid, Extended_Supertoroid, \
                                          Semi_Sphere_Shell, \
                                          Deformed_Superquadric
import trimesh

# ground truth
def mesh_from_primitives(y, split=None, dtype='o3d', info_types_from = 'cfg', **cfg):

    # initialize
    total_pointclouds = []
    total_faces = []
    total_mesh = []
    if info_types_from == 'cfg':
        info_types = cfg['info_types']
        n_types = cfg['n_types']
    else:
        info_types = [info_types_from]
        n_types = 1
    full_params = cfg['full_params']
    full_num_params = cfg['full_num_params']
    full_types = full_num_params.keys()

    for element_batch in range(np.shape(y)[0]):
        
        # load each element in batch
        y_element = y[element_batch, :, :]
        trimesh_total_primitives_mesh = []

        for index in range(np.shape(y_element)[0]):
            
            # primitive information
            primitive_info = y_element[index, :]

            if (split == 'groundtruth' and primitive_info[-1] == 1) or split == 'trained':

                # primitive type
                type_primitive = int(np.argmax(primitive_info[:n_types]))
                type_primitive = info_types[type_primitive]

                # primitive pose
                se3_primitive = primitive_info[n_types:n_types + 12]
                # SE3 = np.array([np.append(se3_primitive[0:3], se3_primitive[9:10]),
                #                 np.append(se3_primitive[3:6], se3_primitive[10:11]), 
                #                 np.append(se3_primitive[6:9], se3_primitive[11:12]), 
                #                 [0, 0, 0, 1]])
                SE3 = np.array([np.append(se3_primitive[3:6], se3_primitive[0:1]),
                                np.append(se3_primitive[6:9], se3_primitive[1:2]), 
                                np.append(se3_primitive[9:12], se3_primitive[2:3]), 
                                [0, 0, 0, 1]])

                # primitive parameters
                parameters_primitive = primitive_info[n_types + 12:] 
                parameters = dict()
                index_param = 0
                for type_ in full_types:
                    if type_ in info_types:
                        if type_primitive == type_:
                            params = full_params[type_primitive]
                            for param_ in params:
                                parameters[param_] = parameters_primitive[index_param]
                                index_param += 1
                        else:
                            index_param += full_num_params[type_]   

                # determine open3d object
                if type_primitive == 'box':
                    primitive = Box(SE3, parameters)
                elif type_primitive == 'cone':
                    primitive = Cone(SE3, parameters)
                elif type_primitive == 'cylinder':
                    primitive = Cylinder(SE3, parameters)
                elif type_primitive == 'sphere':
                    primitive = Sphere(SE3, parameters)
                elif type_primitive == 'torus':
                    primitive = Torus(SE3, parameters)
                elif type_primitive == 'rectangle_ring':
                    primitive = RectangleRing(SE3, parameters)
                elif type_primitive == 'cylinder_ring':
                    primitive = CylinderRing(SE3, parameters)
                elif type_primitive == 'semi_sphere_shell':
                    primitive = Semi_Sphere_Shell(SE3, parameters)
                elif type_primitive == 'superquadric':
                    primitive = Superquadric(SE3, parameters)
                elif type_primitive == 'extended_superquadric':
                    primitive = Extended_Superquadric(SE3, parameters)
                elif type_primitive == 'supertoroid':
                    primitive = Supertoroid(SE3, parameters)
                elif type_primitive == 'extended_supertoroid':
                    primitive = Extended_Supertoroid(SE3, parameters)
                elif type_primitive == 'deformed_superquadric':
                    primitive = Deformed_Superquadric(SE3, parameters)
                    # print(parameters)
                else:
                    print('invalid primitive type!!')

                if index == 0:
                    total_primitives_mesh = primitive.mesh
                else:
                    total_primitives_mesh = total_primitives_mesh + primitive.mesh

                if dtype == 'numpy':
                    pointclouds = np.asarray(total_primitives_mesh.vertices).tolist()
                    faces = np.asarray(total_primitives_mesh.triangles).tolist()

                if dtype == 'trimesh':
                    trimesh_total_primitives_mesh.append(trimesh.Trimesh(vertices=np.asarray(primitive.mesh.vertices), faces=np.asarray(primitive.mesh.triangles)))

        # collect data for batch 
        if dtype == 'numpy':       
            total_pointclouds.append(pointclouds)
            total_faces.append(faces)
        elif dtype == 'o3d':
            total_mesh.append(total_primitives_mesh)
        elif dtype == 'trimesh':
            if len(trimesh_total_primitives_mesh) == 1:
                total_mesh.append(trimesh_total_primitives_mesh[0])
            else:
                total_mesh.append(trimesh.boolean.union(trimesh_total_primitives_mesh, engine='blender'))

    if dtype == 'numpy':
        return np.asarray(total_pointclouds), np.asarray(total_faces)
    elif dtype == 'o3d' or dtype == 'trimesh':
        return total_mesh
    else:
        raise ValueError('check the dtype of mesh_from_primitives function: numpy or o3d')

def meshs_to_numpy(mesh1, mesh2, change_color=True, color1 = [0, 1, 1], color2 = [0, 0, 1], bcoordinate = False):
    
    if len(mesh1) is not len(mesh2):
        raise ValueError('mesh1 and mesh2 do not have same batch size')
    
    total_pointclouds = []
    total_faces = []
    total_colors = []

    max_num_pointclouds = 0
    max_num_faces = 0

    for batch in range(len(mesh1)):
        
        # # color define
        # mesh1_vertices = np.asarray(mesh1[batch].vertices)
        # mesh2_vertices = np.asarray(mesh2[batch].vertices)
        # gt_colors = np.zeros(mesh1_vertices.shape)
        # gt_colors[:,1] = 255
        # train_colors = np.zeros(mesh2_vertices.shape)
        # train_colors[:,2] = 0
        # colors = np.concatenate((gt_colors, train_colors), axis=1)

        # color painting
        if change_color:
            mesh1[batch].paint_uniform_color(color1)
            mesh2[batch].paint_uniform_color(color2)

        # make coordinate frame
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.4, origin=[0, 0, 0]
        )

        # combine meshes
        if bcoordinate:
            mesh = mesh1[batch] + mesh2[batch] + coordinate
        else:
            mesh = mesh1[batch] + mesh2[batch]

        # mesh = mesh1[batch]
        pointclouds = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        colors = 255 * np.asarray(mesh.vertex_colors)

        # append for batch
        total_pointclouds.append(pointclouds)
        total_faces.append(faces)
        total_colors.append(colors)

        # find the maximum dimension
        if pointclouds.shape[0] > max_num_pointclouds:
            max_num_pointclouds = pointclouds.shape[0]
        if faces.shape[0] > max_num_faces:
            max_num_faces = faces.shape[0]

    # matching dimension for tensorboard
    for batch in range(len(mesh1)):
        diff_num_pointclouds = max_num_pointclouds - total_pointclouds[batch].shape[0]
        diff_num_faces = max_num_faces - total_faces[batch].shape[0]
        total_pointclouds[batch] = np.concatenate((total_pointclouds[batch], np.zeros((diff_num_pointclouds, 3))), axis=0)
        total_faces[batch] = np.concatenate((total_faces[batch], np.zeros((diff_num_faces, 3))), axis=0)
        total_colors[batch] = np.concatenate((total_colors[batch], np.zeros((diff_num_pointclouds, 3))), axis=0)

    return np.asarray(total_pointclouds), np.asarray(total_faces), np.asarray(total_colors)

def meshs_to_numpy_with_iou(mesh1, mesh2, iou, color1 = [0, 1, 1], color2 = [0, 0, 1]):
    
    gt_and_val_meshes_verties, gt_and_val_meshes_faces, gt_and_val_meshes_colors = meshs_to_numpy(mesh1, mesh2)
    
    gt_and_val_meshes = []
    iou_meshes = []
    for batch in range(len(mesh1)):
        gt_and_val_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(gt_and_val_meshes_verties[batch]), triangles=o3d.utility.Vector3iVector(gt_and_val_meshes_faces[batch]))
        gt_and_val_mesh.vertex_colors = o3d.utility.Vector3dVector(gt_and_val_meshes_colors[batch] / 255)
        gt_and_val_meshes.append(gt_and_val_mesh)

        iou_pcd = text_3d(str(iou[batch]), [0, 0, -0.8], direction=(1, 0, 0), font_size=5, density=40)
        iou_pcd.estimate_normals()
        iou_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(iou_pcd, o3d.utility.DoubleVector([0.0015]))
        iou_mesh.paint_uniform_color([0, 0, 0])
        iou_meshes.append(iou_mesh)

    return meshs_to_numpy(gt_and_val_meshes, iou_meshes, change_color=False)


def text_3d(text, pos, direction=None, degree=0.0, font='/usr/share/fonts/truetype/freefont/FreeSans.ttf', font_size=16, density=2):

    if direction is None:
        direction = (0., 0., 1.)

    font_obj = ImageFont.truetype(font, font_size * density)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd

def o3d_to_trimesh(o3d_mesh):
    return trimesh.Trimesh(vertices=np.asarray(o3d_mesh.vertices), faces=np.asarray(o3d_mesh.triangles))