import numpy as np
import random
import open3d as o3d
from training.functions.primitives import Cylinder, Box, Sphere, Cone, Torus, Supertoroid, Semi_Sphere_Shell, RectangleRing, CylinderRing, gen_primitive
from training.functions.make_mesh import o3d_to_trimesh
# import util as ut
import json
import training.functions.lie_alg
from scipy.stats import special_ortho_group
import trimesh
# import fcl

class Object_trimesh:
    def __init__(self, set_of_primitives, transform=True, collisionBox=False):
        self.initialzier(set_of_primitives, transform)

    def initialzier(self, set_of_primitives, transform=True):
        self.num_primitives = len(set_of_primitives)
        self.primitives = set_of_primitives.copy()
        self.mesh = o3d_to_trimesh(self.primitives[0].mesh)
        for primitive in self.primitives[1:]:
            self.mesh = trimesh.boolean.union((self.mesh, o3d_to_trimesh(primitive.mesh)), engine='blender')
        if transform is not False:
            self.transform_to_object_frame(transform)

    def transform_object(self, SE3):
        # for primitive in range(self.num_primitives):
        #     self.primitives[primitive].mesh.transform(SE3)
        #     self.primitives[primitive].SE3 = np.matmul(SE3, self.primitives[primitive].SE3)
        self.mesh.apply_transform(SE3)

class Object:
    def __init__(self, set_of_primitives, transform=True, collisionBox=False):
        self.initialzier(set_of_primitives, transform)
        if collisionBox:
            self.collisionBox = []
            for prim in set_of_primitives:
                self.collisionBox.append(prim.collisionBox)

    def initialzier(self, set_of_primitives, transform=True):
        self.num_primitives = len(set_of_primitives)
        self.primitives = set_of_primitives.copy()
        self.mesh = self.primitives[0].mesh
        for primitive in self.primitives[1:]:
            self.mesh = self.mesh + primitive.mesh
        if transform is not False:
            self.transform_to_object_frame(transform)

    def get_object_frame(self):
        num_pts = 10000
        obj_pcd = self.mesh.sample_points_uniformly(number_of_points=num_pts)
        mean, cov = obj_pcd.compute_mean_and_covariance()
        eig_values, eig_vectors = np.linalg.eig(cov)
        principle_axes = eig_vectors[np.argsort(-1 * eig_values), :]
        z = principle_axes[0, :] / np.linalg.norm(principle_axes[0, :])
        x = principle_axes[1, :] / np.linalg.norm(principle_axes[1, :])
        y = np.cross(z, x)
    
        T = np.identity(4)
        T[0:3, 0:3] = np.array([x, y, z]).transpose()
        T[0:3, 3] = mean
        
        return T

    def get_object_frame2(self):
        num_pts = 10000
        obj_pcd = self.mesh.sample_points_uniformly(number_of_points=num_pts)
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(obj_pcd.points)
        if np.abs(np.linalg.det(bbox.R) + 1) < 1e-4:
            object_frame = lie_alg.define_SE3(-bbox.R, bbox.center)
        else:
            object_frame = lie_alg.define_SE3(bbox.R, bbox.center)
        T_xyz_to_zxy = lie_alg.define_SE3(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), [0, 0, 0])
        object_frame = np.matmul(object_frame, T_xyz_to_zxy)
        
        return object_frame

    def transform_object(self, SE3):
        for primitive in range(self.num_primitives):
            self.primitives[primitive].mesh.transform(SE3)
            self.primitives[primitive].SE3 = np.matmul(SE3, self.primitives[primitive].SE3)
            if hasattr(self.primitives[primitive], 'collisionBox'):
                T = lie_alg.define_SE3(self.primitives[primitive].collisionBox.getRotation(), self.primitives[primitive].collisionBox.getTranslation())
                T = np.dot(SE3, T)
                self.primitives[primitive].collisionBox.setTransform(fcl.Transform(lie_alg.get_SO3(T), lie_alg.get_p(T)))
        self.mesh.transform(SE3)

    def transform_to_object_frame(self,transform):
        object_frame = self.get_object_frame2()
        if transform == 'center':
            object_frame = lie_alg.change_SO3(object_frame, np.identity(3))
        inv_obj_frame = lie_alg.inverse_SE3(object_frame)
        self.transform_object(inv_obj_frame) 

    def add_to_vis(self,vis):
        vis.add_geometry(self.mesh)


class Box_Object(Object):

    def __init__(self, config):
        # config requires 'random' boolean. If 'random' is True, w min max, d min max, h min max are required.
        if config['random'] is True:
            w = random.uniform(config['w']['min'], config['w']['max'])
            d = random.uniform(config['d']['min'], config['d']['max'])
            h = random.uniform(config['h']['min'], config['h']['max'])

        else: 
            w = config['w']
            d = config['d']
            h = config['h']

        lengths = [w, d, h]
        lengths.sort()

        w = lengths[0]
        d = lengths[1]
        h = lengths[2]

        P1_parameters = {"width": w, "depth": d, "height": h}

        P1_SE3 = np.identity(4)

        if 'color' in config.keys():
            P1_color = config['color']
            P1 = Box(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
        else:
            P1 = Box(SE3=P1_SE3, parameters=P1_parameters, color=None)

        self.initialzier([P1], transform=False)


class Cylinder_Object(Object):

    def __init__(self, config):
        # config requires 'random' boolean. If 'random' is True, r min max, h min max are required.
        # if config['random'] is True:
        # 	if random.random() < 0.3:
        # 		r = random.uniform(config['r']['min'], config['r']['max'])
        # 		h = random.uniform(r * 1, r * 2.3)
        # 	else:
        # 		r = random.uniform(config['r']['min'], config['r']['max'])
        # 		h = random.uniform(r * 2.3, r * 20)
                

        # else: 
        # 	r = config['r']
        # 	h = config['h']

        if config['random'] is True:
            r = random.uniform(config['r']['min'], config['r']['max'])
            h = random.uniform(config['h']['min'], config['h']['max'])
        else: 
            r = config['r']
            h = config['h']

        P1_SE3 = np.identity(4)
        P1_parameters = {"radius": r, "height": h}

        if 'color' in config.keys():
            P1_color = config['color']
            P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
        else:
            P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters)

        self.initialzier([P1], transform=False)


class Cone_Object(Object):

    def __init__(self, config):
        # config requires 'random' boolean. If 'random' is True, r_u min max, r_l min max, h min max are required.
        if config['random'] is True:
            r_l = random.uniform(config['r_l']['min'], config['r_l']['max'])
            if 'r_u' in config.keys() and config['r_u'] is not None:
                r_u = random.uniform(config['r_u']['min'], min(config['r_u']['max'], 0.8 * r_l))
   
                if r_u > r_l:
                    r_u, r_l = r_l, r_u
                elif r_u == r_l:
                    r_u = r_u - config['r_u']['min']/2
   
            else:
                r_u = 0
            h = random.uniform(config['h']['min'], config['h']['max'])

        else: 
            r_l = config['r_l']
            r_u = config['r_u']
            h = config['h']
        
        P1_SE3 = np.identity(4)
        P1_parameters = {"lower_radius": r_l, "upper_radius": r_u, "height": h}

        if 'color' in config.keys():
            P1_color = config['color']
            P1 = Cone(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
        else:
            P1 = Cone(SE3=P1_SE3, parameters=P1_parameters)

        self.initialzier([P1], transform=False)


class Sphere_Object(Object):

    def __init__(self, config):
        # config requires 'random' boolean. If 'random' is True, r min max, h min max are required.
        if config['random'] is True:
            r = random.uniform(config['r']['min'], config['r']['max'])
            if 'h' in config.keys() and config['h'] is True:
                h = random.uniform(r * 0.1, r * 1.9)
            else:
                h = 0

        else: 
            r = config['r']
            h = config['h']
        
        P1_SE3 = np.identity(4)
        P1_parameters = {"radius": r, "height": h}
        
        if 'color' in config.keys():
            P1_color = config['color']
            P1 = Sphere(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
        else:
            P1 = Sphere(SE3=P1_SE3, parameters=P1_parameters)

        self.initialzier([P1], transform=False)

class Torus_Object(Object):

    def __init__(self, config):
        if config['random'] is True:
            r_torus = random.uniform(config['r_torus']['min'], config['r_torus']['max'])
            r_tube = random.uniform(max(0.01, config['r_tube']['min']), min(config['r_tube']['max'], 0.8 * r_torus))
            if 'c1' in config.keys() and config['c1'] is True:
                c1 = random.uniform(3.14/6, 3.14/2)
            else:
                c1 = r_torus + r_tube
        else: 
            r_torus = config['r_torus']
            r_tube = config['r_tube']
            c1 = config['c1']
    
        P1_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, 0])
        P1_parameters = {"torus_radius": r_torus, "tube_radius": r_tube, "c1": c1}

        if 'color1' in config.keys():
            P1_color = config['color1']
        else:
            P1_color = None

        P1 = Torus(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
    
        self.initialzier([P1], transform=False)

class RectangleRing_Object(Object):

    def __init__(self, config):
        if config['random'] is True:
            d = random.uniform(config['d']['min'], config['d']['max'])
            w = random.uniform(max(d, config['w']['min']), config['w']['max'])
            h = random.uniform(config['h']['min'], config['h']['max'])
            t2 = random.uniform(config['t']['min'], min(config['t']['max'], 0.2 * d))
            t1 = random.uniform(config['t']['min'], min(config['t']['max'], 0.2 * w, w * (d**2 - 2 * d * t2 - t2**2) / (2 * d * (d - 2 * t2))))
            if 'c1' in config.keys() and config['c1'] is True:
                c1 = random.uniform((-w / 2 + t1) * 0.8, 0.9 * w / 2)
            else:
                c1 = w / 2 
        else: 
            w = config['w']
            d = config['d']
            h = config['h']
            t1 = config['t1']		
            t2 = config['t2']
            c1 = config['c1']

        P1_SE3 = np.identity(4)
        # P1_parameters = {'a1': w / d * t / 2  , 'a2': t / 2 , 'a3': h / 2 , 'a4': (d / 2 - t / 2) * 2 / t, 'e1': 0, 'e2': 0}
        # P1_parameters = {'a1': (w / 2 - t1) / (d / t2 - 2)  , 'a2': t2 / 2 , 'a3': h / 2 , 'a4': (d / 2 - t2 / 2) * 2 / t2, 'e1': 0, 'e2': 0}
        # if 'color' in config.keys():
        # 	P_color = config['color']
        # else:
        # 	P_color = None
        # P1 = Supertoroid(SE3=P1_SE3, parameters=P1_parameters, color=P_color)
        # P1.mesh.remove_duplicated_vertices()
        # P1.mesh.remove_degenerate_triangles()
        # vertices = np.asarray(P1.mesh.vertices)
        # x_top_vertices = np.squeeze(np.argwhere(vertices[:,0] == max(vertices[:, 0])))
        # x_bottom_vertices = np.squeeze(np.argwhere(vertices[:,0] == min(vertices[:, 0])))
        # vertices[x_top_vertices, 0] = w / 2
        # vertices[x_bottom_vertices, 0] = - w / 2
        # P1.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        P1_parameters = {"width": w, "depth": d, "height": h, "thickness1": t1, "thickness2": t2, "c1": c1}

        if 'color1' in config.keys():
            P1_color = config['color1']
        else:
            P1_color = None

        P1 = RectangleRing(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)

        self.initialzier([P1], transform=False)

class CylinderRing_Object(Object):

    def __init__(self, config):
        if config['random'] is True:
            r = random.uniform(config['r']['min'], config['r']['max'])
            h = random.uniform(config['h']['min'], config['h']['max'])
            t = random.uniform(config['t']['min'], min(config['t']['max'], 0.2 * r))
            if 'c1' in config.keys() and config['c1'] is True:
                c1 = random.uniform(0.8 * (-r + t), 0.8 * (r - t))
            else:
                c1 = r

        else: 
            r = config['r']
            h = config['h']
            t = config['t']	
            c1 = config['c1']	
        
        P1_SE3 = np.identity(4)
        P1_parameters = {'radius_outer': r, 'radius_inner': r - t, 'height': h, 'c1': c1}
        if 'color' in config.keys():
            P_color = config['color']
        else:
            P_color = None
        P1 = CylinderRing(SE3=P1_SE3, parameters=P1_parameters, color=P_color)

        self.initialzier([P1], transform=False)


class SemiSphereShell_Object(Object):

    def __init__(self, config):
        if config['random'] is True:
            t = random.uniform(config['t']['min'], config['t']['max'])
            r = random.uniform(max(config['r']['min'], t + 0.02), config['r']['max'])
        else: 
            r = config['r']
            r = config['t']
    
        P1_SE3 = np.identity(4)
        P1_parameters = {"radius_outer": r, "radius_inner": r - t}

        if 'color1' in config.keys():
            P1_color = config['color1']
        else:
            P1_color = None

        P1 = Semi_Sphere_Shell(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
    
        self.initialzier([P1], transform=False)

class Pencil(Object):

    def __init__(self, config):
        # config requires 'random' boolean. If 'random' is True, h1 min max, h2 min max, h3 min max, r1 min max, r_u2 min max, r3 max are required.
        if config['random'] is True:
            h1 = random.uniform(config['h1']['min'], config['h1']['max'])
            h2 = random.uniform(config['h2']['min'], config['h2']['max'])

            r1 = random.uniform(config['r1']['min'], config['r1']['max'])
            r_l2 = r1
            r_u2 = 0
        else: 
            h1 = config['h1']
            h2 = config['h2']
            h3 = config['h3']

            r1 = config['r1']
            r_l2 = config['r_l2']
            r_u2 = config['r_u2']

        P1_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, 0])
        P2_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, h1 / 2 + h2 / 2])

        P1_parameters = {"radius": r1, "height": h1}
        P2_parameters = {"lower_radius": r_l2, "upper_radius": r_u2, "height": h2}

        if 'color1' in config.keys():
            P1_color = config['color1']
            P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
        else:
            P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters)
        if 'color2' in config.keys():
            P2_color = config['color2']
            P2 = Cone(SE3=P2_SE3, parameters=P2_parameters, color=P2_color)
        else:
            P2 = Cone(SE3=P2_SE3, parameters=P2_parameters)
        
    
        self.initialzier([P1, P2], transform='center')

class Bottle_Cone(Object):

    def __init__(self, config, collisionBox=False):
        # config requires 'random' boolean. If 'random' is True, h1 min max, h2 min max, h3 min max, r1 min max, r_u2 min max, r3 max are required.
        if config['random'] is True:
            h1 = random.uniform(config['h1']['min'], config['h1']['max'])
            h2 = random.uniform(config['h2']['min'], config['h2']['max'])
            h3 = random.uniform(config['h3']['min'], config['h3']['max'])

            r1 = random.uniform(config['r1']['min'], config['r1']['max'])
            r_l2 = r1
            r_u2 = random.uniform(config['r_u2']['min'], min(config['r_u2']['max'], 2 * r_l2 / 3))
            r3 = random.uniform(r_u2, min(config['r3']['max'], r1-0.001))
        else: 
            h1 = config['h1']
            h2 = config['h2']
            h3 = config['h3']

            r1 = config['r1']
            r_l2 = config['r_l2']
            r_u2 = config['r_u2']
            r3 = config['r3']

        P1_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, 0])
        P2_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, h1 / 2 + h2 / 2])
        P3_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, h1 / 2 + h2 + h3 / 2])

        P1_parameters = {"radius": r1, "height": h1}
        P2_parameters = {"lower_radius": r_l2, "upper_radius": r_u2, "height": h2}
        P3_parameters = {"radius": r3, "height": h3}

        if 'color1' in config.keys():
            P1_color = config['color1']
            P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, color=P1_color, collisionBox=collisionBox)
        else:
            P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, collisionBox=collisionBox)
        if 'color2' in config.keys():
            P2_color = config['color2']
            P2 = Cone(SE3=P2_SE3, parameters=P2_parameters, color=P2_color, collisionBox=collisionBox )
        else:
            P2 = Cone(SE3=P2_SE3, parameters=P2_parameters, collisionBox=collisionBox)
        if 'color3' in config.keys():
            P3_color = config['color3']
            P3 = Cylinder(SE3=P3_SE3, parameters=P3_parameters, color=P3_color, collisionBox=collisionBox)
        else:
            P3 = Cylinder(SE3=P3_SE3, parameters=P3_parameters, collisionBox=collisionBox)
    
        self.initialzier([P1, P2, P3], transform='center')
        if collisionBox:
            self.collisionBox = []
            for primitive in self.primitives:
                self.collisionBox.append(primitive.collisionBox)


class Bottle_Sphere(Object):

    def __init__(self, config):
        # config requires 'random' boolean. If 'random' is True, h1 min max, h2 min max, h3 min max, r1 min max, r_u2 min max, r3 max are required.
        if config['random'] is True:
            r1 = random.uniform(config['r1']['min'], config['r1']['max'])
            h1 = random.uniform(config['h1']['min'], config['h1']['max'])

            h2 = random.uniform(r1, 3 * r1)
            r2 = (r1 * r1 + h2 * h2) / (2 * h2)

            r3 = random.uniform(config['r3']['min'], min(config['r3']['max'], r1 - 0.005))
            h3 = random.uniform(config['h3']['min'], config['h3']['max'])
            
            if config['cap']:
                z3 = random.uniform(h1 / 2 + 2 * r2 - h2 - h3 / 2, h1/ 2 + 2 * r2 -h2 + h3 / 3)
            else:
                if random.random() < 0.3:
                    h2 = r1
                    r2 = r1
                z3 = h1 / 2 + r2 - h2 + np.sqrt(r2 * r2 - r3 * r3) + h3 / 2
        else: 
            h1 = config['h1']
            h2 = config['h2']
            h3 = config['h3']

            r1 = config['r1']
            r2 = config['r2']
            r3 = config['r3']

            z3 = config['z3']

        P1_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, 0])
        P2_SE3 = lie_alg.define_SE3(lie_alg.exp_so3(np.array([1, 0, 0]) * np.pi), [0, 0, h1 / 2 - h2 + r2])
        P3_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, z3])

        P1_parameters = {"radius": r1, "height": h1}
        P2_parameters = {"radius": r2, "height": h2}
        P3_parameters = {"radius": r3, "height": h3}

        if 'color1' in config.keys():
            P1_color = config['color1']
        else:
            P1_color = None
        if 'color2' in config.keys():
            P2_color = config['color2']
        else:
            P2_color = None
        if 'color3' in config.keys():
            P3_color = config['color3']
        else:
            P3_color = None

        P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
        P2 = Sphere(SE3=P2_SE3, parameters=P2_parameters, color=P2_color)
        P3 = Cylinder(SE3=P3_SE3, parameters=P3_parameters, color=P3_color)
    
        self.initialzier([P1, P2, P3], transform='center')

class Mic(Object):

    def __init__(self, config):
        # config requires 'random' boolean. If 'random' is True, h1 min max, h2 min max, h3 min max, r1 min max, r_u2 min max, r3 max are required.
        if config['random'] is True:
            r1 = random.uniform(config['r1']['min'], config['r1']['max'])
            h1 = random.uniform(config['h1']['min'], config['h1']['max'])

            h2 = random.uniform(r1 / np.sqrt(15), r1 / np.sqrt(3))
            r2 = (r1 * r1 + h2 * h2) / (2 * h2)
        else: 
            h1 = config['h1']
            h2 = config['h2']

            r1 = config['r1']
            r2 = config['r2']


        P1_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, 0])
        P2_SE3 = lie_alg.define_SE3(lie_alg.exp_so3(np.array([1, 0, 0]) * np.pi), [0, 0, h1 / 2 - h2 + r2])

        P1_parameters = {"radius": r1, "height": h1}
        P2_parameters = {"radius": r2, "height": h2}

        if 'color1' in config.keys():
            P1_color = config['color1']
        else:
            P1_color = None
        if 'color2' in config.keys():
            P2_color = config['color2']
        else:
            P2_color = None


        P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
        P2 = Sphere(SE3=P2_SE3, parameters=P2_parameters, color=P2_color)
    
        self.initialzier([P1, P2], transform='center')

class Can(Object):

    def __init__(self, config):
        # config requires 'random' boolean. If 'random' is True, h1 min max, h2 min max, h3 min max, r1 min max, r_u2 min max, r3 max are required.
        if config['random'] is True:
            h1 = random.uniform(config['h1']['min'], config['h1']['max'])
            h2 = random.uniform(config['h2']['min'], config['h2']['max'])
            h3 = random.uniform(config['h3']['min'], config['h3']['max'])

            r2 = random.uniform(config['r2']['min'], config['r2']['max'])
            r_l1 = r2
            r_l3 = r2
            r_u1 = random.uniform(0.7 * r2, 0.9 * r2)
            r_u3 = random.uniform(max(0.7 * r2, r_u1), r_u1)
        else: 
            h1 = config['h1']
            h2 = config['h2']
            h3 = config['h3']

            r_u1 = config['r_u1']
            r_l1 = config['r_l1']
            r2 = config['r2']
            r_u3 = config['r_u3']
            r_l3 = config['r_l3']

        P1_SE3 = lie_alg.define_SE3(lie_alg.exp_so3(np.array([0, 1, 0]) * np.pi), [0, 0, 0])
        P2_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, h1 / 2 + h2 / 2])
        P3_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, h1 / 2 + h2 + h3 / 2])

        P1_parameters = {"lower_radius": r_l1, "upper_radius": r_u1, "height": h1}
        P2_parameters = {"radius": r2, "height": h2}
        P3_parameters = {"lower_radius": r_l3, "upper_radius": r_u3, "height": h3}

        if 'color1' in config.keys():
            P1_color = config['color1']
            P1 = Cone(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
        else:
            P1 = Cone(SE3=P1_SE3, parameters=P1_parameters)
        if 'color2' in config.keys():
            P2_color = config['color2']
            P2 = Cylinder(SE3=P2_SE3, parameters=P2_parameters, color=P2_color)
        else:
            P2 = Cylinder(SE3=P2_SE3, parameters=P2_parameters)
        if 'color3' in config.keys():
            P3_color = config['color3']
            P3 = Cone(SE3=P3_SE3, parameters=P3_parameters, color=P3_color)
        else:
            P3 = Cone(SE3=P3_SE3, parameters=P3_parameters)
    
        self.initialzier([P1, P2, P3], transform='center')


class Cup(Object):

    def __init__(self, config):
        # config requires 'random' boolean. If 'random' is True, h1 min max, h2 min max, h3 min max, r1 min max, r_u2 min max, r3 max are required.
        if config['random'] is True:
            r1 = random.uniform(config['r1']['min'], config['r1']['max'])
            h1 = random.uniform(config['h1']['min'], config['h1']['max'])
            t1 = random.uniform(config['t1']['min'], min(config['t1']['max'], 0.2 * r1))

            h2 = random.uniform(config['h2']['min'], config['h2']['max'])

            r_tube = random.uniform(config['r_tube']['min'], config['r_tube']['max'])
            r_torus = random.uniform(config['r_torus']['min'], min(config['r_torus']['max'], h1 / 2 - r_tube))
            c1 = random.uniform(0.3 * (-r_torus + r_tube), r_torus)
            z3 = random.uniform(-h1 / 2 + (r_tube + r_torus), h1 / 2 - (r_tube + r_torus))
            if random.random() < 0.3:
                z3 = 0 
        else: 
            r1 = config['r1']
            h1 = config['h1']
            t1 = config['t1']

            h2 = config['h2']

            r_tube = config['r_tube']
            r_torus = config['r_torus']
            c1 = config['c1']
            z3 = config['z3']
        
        r2 = r1

        P1_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, 0])
        P2_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, - h1 / 2 - h2 / 2])
        P3_SE3 = lie_alg.define_SE3(np.dot(lie_alg.exp_so3(np.array([1, 0, 0]) * np.pi / 2), lie_alg.exp_so3(np.array([0, 0, 1]) * np.pi)), [r1 - t1 / 2 + c1, 0, z3])
        
        P1_parameters = {"radius_outer": r1, "radius_inner": r1 - t1, "height": h1}
        P2_parameters = {"radius": r2,"height": h2}
        P3_parameters = {"torus_radius": r_torus, "tube_radius": r_tube, "c1": c1}
        
        if 'color1' in config.keys():
            P1_color = config['color1']
        else:
            P1_color = None
        if 'color2' in config.keys():
            P2_color = config['color2']
        else:
            P2_color = None
        if 'color3' in config.keys():
            P3_color = config['color3']
        else:
            P3_color = None

        P1 = CylinderRing(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
        P2 = Cylinder(SE3=P2_SE3, parameters=P2_parameters, color=P2_color)
        P3 = Torus(SE3=P3_SE3, parameters=P3_parameters, color=P3_color)

        self.initialzier([P1, P2, P3], transform=False)	

class Dumbbell(Object):

    def __init__(self, config):
        # config requires 'random' boolean. If 'random' is True, h1 min max, h2 min max, h3 min max, r1 min max, r_u2 min max, r3 max are required.
        if config['random'] is True:
            h1 = random.uniform(config['h1']['min'], config['h1']['max'])
            h2 = random.uniform(config['h2']['min'], config['h2']['max'])

            r2 = random.uniform(config['r2']['min'], config['r2']['max'])
            r1 = random.uniform(max(config['r1']['min'], r2), config['r1']['max'])
        else: 
            h1 = config['h1']
            h2 = config['h2']

            r1 = config['r1']
            r2 = config['r2']
        
        r3 = r1
        h3 = h1

        P1_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, 0])
        P2_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, h1 / 2 + h2 / 2])
        P3_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, h1 / 2 + h2 + h3 / 2])

        P1_parameters = {"radius": r1, "height": h1}
        P2_parameters = {"radius": r2, "height": h2}
        P3_parameters = {"radius": r3, "height": h3}

        if 'color1' in config.keys():
            P1_color = config['color1']
        else:
            P1_color = None
        if 'color2' in config.keys():
            P2_color = config['color2']
        else:
            P2_color = None
        if 'color3' in config.keys():
            P3_color = config['color3']
        else:
            P3_color = None

        P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
        P2 = Cylinder(SE3=P2_SE3, parameters=P2_parameters, color=P2_color)
        P3 = Cylinder(SE3=P3_SE3, parameters=P3_parameters, color=P3_color)
    
        self.initialzier([P1, P2, P3], transform='center')

class Hammer_Cylinder(Object):

    def __init__(self, config):
        # config requires 'random' boolean. If 'random' is True, h1 min max, h2 min max, h3 min max, r1 min max, r_u2 min max, r3 max are required.
        if config['random'] is True:
            r1 = random.uniform(config['r1']['min'], config['r1']['max'])
            r2 = random.uniform(max(config['r2']['min'], r1), config['r2']['max'])
        
            h2 = random.uniform(config['h2']['min'], config['h2']['max'])
            if config['z2_offset'] and random.random() < 0.5:
                z2_offset = random.uniform(r2, r2 + 0.010)
                h1 = random.uniform(config['h1']['min'], config['h1']['max']) + z2_offset + r2
            else:
                z2_offset = 0
                h1 = random.uniform(config['h1']['min'], config['h1']['max'])


        else: 
            h1 = config['h1']
            h2 = config['h2']

            r1 = config['r1']
            r2 = config['r2']
            z2_offset = config['z2_offset']
        
        P1_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, 0])
        P2_SE3 = lie_alg.define_SE3(lie_alg.exp_so3(np.array([1, 0 ,0]) * np.pi / 2), [0, 0, h1 / 2 - z2_offset])

        P1_parameters = {"radius": r1, "height": h1}
        P2_parameters = {"radius": r2, "height": h2}

        if 'color1' in config.keys():
            P1_color = config['color1']
        else:
            P1_color = None
        if 'color2' in config.keys():
            P2_color = config['color2']
        else:
            P2_color = None

        P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
        P2 = Cylinder(SE3=P2_SE3, parameters=P2_parameters, color=P2_color)
    
        self.initialzier([P1, P2], transform='center')


class Hammer_Box(Object):

    def __init__(self, config):
        # config requires 'random' boolean. If 'random' is True, h1 min max, h2 min max, h3 min max, r1 min max, r_u2 min max, r3 max are required.
        if config['random'] is True:
            r1 = random.uniform(config['r1']['min'], config['r1']['max'])

            w2 = random.uniform(max(config['w2']['min'], 2 * r1), config['w2']['max'])
            d2 = random.uniform(max(config['d2']['min'], w2), config['d2']['max'])
            h2 = random.uniform(max(config['h2']['min'], d2), config['h2']['max'])

            if config['z2_offset']:
                z2_offset = random.uniform(-d2 / 2, d2 / 2 + 0.010)
            else:
                z2_offset = - d2 / 2

            h1 = random.uniform(config['h1']['min'], config['h1']['max']) + d2 / 2 + z2_offset
            
        else:
            r1 = config['r1'] 
            h1 = config['h1']

            w2 = config['w2']
            d2 = config['d2']
            h2 = config['h2']

            z2_offset = config['z2_offset']
        
        P1_SE3 = lie_alg.define_SE3(np.identity(3), [0, 0, 0])
        P2_SE3 = lie_alg.define_SE3(lie_alg.exp_so3(np.array([1, 0 ,0]) * np.pi / 2), [0, 0, h1 / 2 - z2_offset])

        P1_parameters = {"radius": r1, "height": h1}
        P2_parameters = {"width": w2, "depth": d2, "height": h2}

        if 'color1' in config.keys():
            P1_color = config['color1']
        else:
            P1_color = None
        if 'color2' in config.keys():
            P2_color = config['color2']
        else:
            P2_color = None

        P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
        P2 = Box(SE3=P2_SE3, parameters=P2_parameters, color=P2_color)
    
        self.initialzier([P1, P2], transform='center')

class Randomization(Object):

    def __init__(self, config):
        set_of_prims = [] 
        prim_centers = []
        prim_ext = []
        for prim in range(config['num_max_prims']):
            if prim > 1 and random.random() < 0.3:
                continue

            prob = random.random()
            if prob < 0.25:
                primitive_type = 'box'
            elif prob < 0.5:
                primitive_type = 'cylinder'
            elif prob < 0.75:
                primitive_type = 'cone'
            else:
                primitive_type = 'sphere'
        

            if primitive_type == 'box':
                w = random.uniform(config['box']['w']['min'], config['box']['w']['max'])
                d = random.uniform(config['box']['d']['min'], config['box']['d']['max'])
                h = random.uniform(config['box']['h']['min'], config['box']['h']['max'])

                lengths = sorted([w, d, h])

                w = lengths[0]
                d = lengths[1]
                h = lengths[2]

                P_parameters = {"width": w, "depth": d, "height": h}
                prim_ext.append(h / 2)

            if primitive_type == 'cone':
                r_l = random.uniform(config['cone']['r_l']['min'], config['cone']['r_l']['max'])
                if random.random() <= 0.5:
                    # truncated
                    r_u = random.uniform(min(config['cone']['r_u']['min'], 0.2 * r_l), min(config['cone']['r_u']['max'], 0.7 * r_l))
                    if r_u > r_l:
                        r_u, r_l = r_l, r_u
                    elif r_u == r_l:
                        r_u = r_u - config['cone']['r_u']['min']/2
                else:
                    r_u = 0
                h = random.uniform(config['cone']['h']['min'], config['cone']['h']['max'])

                P_parameters = {"lower_radius": r_l, "upper_radius": r_u, "height": h}
                prim_ext.append(max(h / 2, r_l))

            if primitive_type == 'cylinder':
                r = random.uniform(config['cylinder']['r']['min'], config['cylinder']['r']['max'])
                h = random.uniform(config['cylinder']['h']['min'], config['cylinder']['h']['max'])

                P_parameters = {"radius": r, "height": h}
                prim_ext.append(max(h / 2, r))
            
            if primitive_type == 'sphere':
                r = random.uniform(config['sphere']['r']['min'], config['sphere']['r']['max'])
                if random.random() < 0.5:
                    # truncated
                    h = random.uniform(r * 0.1, r * 1.9)
                else:
                    h = 0
            
                P_parameters = {"radius": r, "height": h}
                prim_ext.append(r)

            while True:
                p1 = random.uniform(-config['work_space_size'], config['work_space_size'])
                p2 = random.uniform(-config['work_space_size'], config['work_space_size'])
                p3 = random.uniform(-config['work_space_size'], config['work_space_size'])
                p = np.array([p1, p2, p3])
                is_far_enough = True
                for i, prim_center in enumerate(prim_centers):
                    if np.linalg.norm(prim_center-p) < prim_ext[i] + prim_ext[-1]:
                        is_far_enough = False
                        break
                
                if is_far_enough:
                    prim_centers.append(p)
                    break

            P_SE3 = lie_alg.define_SE3(special_ortho_group.rvs(3), p)
            P = gen_primitive[primitive_type](SE3=P_SE3, parameters=P_parameters, color=config['colors'][prim])
            set_of_prims.append(P)


        self.initialzier(set_of_prims, transform=False)


class Drill(Object):

    def __init__(self, config):
        # config requires 'random' boolean. If 'random' is True, h1 min max, h2 min max, h3 min max, r1 min max, r_u2 min max, r3 max are required.
        if config['random'] is True:
            r1 = random.uniform(config['r1']['min'], config['r1']['max'])
            h1 = random.uniform(config['h1']['min'], config['h1']['max'])

            r2 = random.uniform(r1* 0.8, r1)
            h2 = random.uniform(config['h2']['min'], min(config['h2']['max'], 0.8 * h1))

            r_u3 = random.uniform(config['r_u3']['min'], 0.9 * r2)
            h3 = random.uniform(config['h3']['min'], min(config['h3']['max'], 0.5 * h2))

            w4 = random.uniform(config['w4']['min'], min(config['w4']['max'], 1.2 * r1))
            d4 = random.uniform(config['d4']['min'], min(config['d4']['max'], 0.8 * h1))
            h4 = random.uniform(config['h4']['min'], config['h4']['max'])
            theta4 = random.uniform(0, np.pi / 6)
            y4 = random.uniform(- (h1 - d4) / 2 * 0.4, (h1 - d4) / 2 * 0.8)

            w5 = random.uniform(w4, w4 * 1.8)
            d5 = random.uniform(1.2 * d4, config['d5']['max'])
            h5 = random.uniform(config['h5']['min'], config['h5']['max'])
            theta5 = random.uniform(0, np.pi / 6)
            offset5 = random.uniform(-0.02, 0.01)

        else:
            r1 = config['r1'] 
            h1 = config['h1']

            r2 = config['r2'] 
            h2 = config['h2']
        
            r_u3 = config['r_u3'] 
            h3 = config['h3'] 

            w4 = config['w4'] 
            d4 = config['d4'] 
            h4 = config['h4'] 
            theta4 = config['theta4'] 
            y4 = config['y4']

            w5 = config['w5'] 
            d5 = config['d5'] 
            h5 = config['h5'] 
            theta5 = config['theta5'] 
            offset5 = config['offeset5']

        r_l3 = r2

        P1_SE3 = lie_alg.define_SE3(lie_alg.exp_so3(np.array([1, 0 ,0]) * np.pi / 2), [0, 0, 0])
        P2_SE3 = lie_alg.define_SE3(lie_alg.exp_so3(np.array([1, 0 ,0]) * np.pi / 2), [0, h1 / 2 + h2 / 2, 0])
        P3_SE3 = lie_alg.define_SE3(lie_alg.exp_so3(np.array([-1, 0 ,0]) * np.pi / 2), [0, h1 / 2 + h2 + h3 / 2, 0])
        P4_SE3 = lie_alg.define_SE3(lie_alg.exp_so3(np.array([-1, 0 ,0]) * theta4), [0, y4 - h4 * np.sin(theta4) / 2, - (np.sqrt(r1**2 - (w4 / 2)**2) + h4 * np.cos(theta4) / 2 - d4 * np.sin(theta4) / 2)])
        P5_SE3 = lie_alg.define_SE3(lie_alg.exp_so3(np.array([-1, 0 ,0]) * theta5), [0, y4 - h4 * np.sin(theta4) - d5 * np.cos(theta5) / 2 - offset5 + d5 / 2, - (np.sqrt(r1**2 - (w4 / 2)**2) + h4 * np.cos(theta4) - d4 * np.sin(theta4) + h5 / 3)])

        P1_parameters = {"radius": r1, "height": h1}
        P2_parameters = {"radius": r2, "height": h2}
        P3_parameters = {"lower_radius": r_l3, "upper_radius": r_u3, "height": h3}
        P4_parameters = {"width": w4, "depth": d4, "height": h4}
        P5_parameters = {"width": w5, "depth": d5, "height": h5}

        if 'color1' in config.keys():
            P1_color = config['color1']
        else:
            P1_color = None
        if 'color2' in config.keys():
            P2_color = config['color2']
        else:
            P2_color = None
        if 'color3' in config.keys():
            P3_color = config['color3']
        else:
            P3_color = None
        if 'color4' in config.keys():
            P4_color = config['color4']
        else:
            P4_color = None
        if 'color5' in config.keys():
            P5_color = config['color5']
        else:
            P5_color = None

        P1 = Cylinder(SE3=P1_SE3, parameters=P1_parameters, color=P1_color)
        P2 = Cylinder(SE3=P2_SE3, parameters=P2_parameters, color=P2_color)
        P3 = Cone(SE3=P3_SE3, parameters=P3_parameters, color=P3_color)
        P4 = Box(SE3=P4_SE3, parameters=P4_parameters, color=P4_color)
        P5 = Box(SE3=P5_SE3, parameters=P5_parameters, color=P5_color)
    
        self.initialzier([P1, P2, P3, P4, P5], transform='center')






load_object = {
    # please add object_class here
    "bottle_cone": Bottle_Cone,
    "bottle_sphere": Bottle_Sphere,
    "bottle_sphere_with_cap": Bottle_Sphere,
    "box": Box_Object,
    "can": Can,
    "cone": Cone_Object,
    "cup": Cup,
    "cylinder": Cylinder_Object,
    "dumbbell": Dumbbell,
    "hammer_cylinder": Hammer_Cylinder,
    "hammer_box": Hammer_Box,
    "sphere": Sphere_Object,
    "mic": Mic,
    "pencil": Pencil,
    "randomization": Randomization,
    "torus": Torus_Object,
    "rectangle_ring": RectangleRing_Object,
    "cylinder_ring": CylinderRing_Object,
    "semi_sphere_shell": SemiSphereShell_Object,
    "drill": Drill,
}

if __name__ == '__main__':
    config = dict(
              random = True,
              r1 = {'min': 0.03, 'max': 0.04},
              h1 = {'min': 0.06, 'max': 0.1},

              h2 = {'min': 0.02, 'max': 0.06},

              r_u3 = {'min': 0.01},
              h3 = {'min': 0.01, 'max': 0.3},

              w4 = {'min': 0.02, 'max': 0.04},
              d4 = {'min': 0.03, 'max': 0.05},
              h4 = {'min': 0.07, 'max': 0.12},
              
              d5 = {'min': 0.08, 'max': 0.12},
              h5 = {'min': 0.03, 'max': 0.05},
              )

    object_name = 'drill'
    # with open(f"data_generation/object_config/{object_name}.json") as readfile:
    # 	config = json.load(readfile)

    obj = load_object[object_name](config)
    # obj.mesh.paint_uniform_color([0.9, 0.9, 0.9])
    frame_SE3 = np.identity(4)
    print(obj.primitives[0].parameters)
    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    ut.add_obj_to_vis(obj, vis)
    #ut.add_frame(frame_SE3, vis, size=0.1)

    vis.run()
    vis.destroy_window()
    print(obj.primitives[0].SE3)

    with open("data_generation/object_config/drill.json", "w") as outfile:  
         json.dump(config, outfile, indent=4)