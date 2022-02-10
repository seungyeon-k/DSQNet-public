import math
import numpy as np
import open3d as o3d
import warnings
import copy
# import lie
# import trimesh

class Base_primitives:

    def transform_mesh(self):
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color(self.color)
        self.mesh.transform(self.SE3)

class Cylinder(Base_primitives):
    def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8]):
        self.type = 'cylinder'
        self.SE3 = SE3
        self.parameters = parameters
        self.color = color
        self.mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=self.parameters['radius'], height=self.parameters['height'], resolution=100)
        self.transform_mesh()

class Box(Base_primitives):

    def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8]):
        self.type = 'box'
        self.SE3 = SE3
        self.parameters = parameters
        self.color = color
        self.mesh = o3d.geometry.TriangleMesh.create_box(width=self.parameters['width'], depth=self.parameters['height'], height=self.parameters['depth']) # change is intentional!
        self.mesh.translate([-self.parameters['width']/2, -self.parameters['depth']/2, -self.parameters['height']/2])
        self.transform_mesh()

class Sphere(Base_primitives):

    def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8], resolution=100):
        self.type = 'sphere'
        self.SE3 = SE3
        self.parameters = parameters
        self.color = color
        self.mesh = o3d.geometry.TriangleMesh.create_sphere(radius = self.parameters['radius'], resolution = resolution)
        if parameters['height'] > 0 and parameters['height'] < 2 * self.parameters['radius']:
            nearset_elevation_index = math.floor(resolution / math.pi * math.acos((self.parameters['radius'] - self.parameters['height']) / self.parameters['radius']))
            nearest_height = math.cos(math.pi * nearset_elevation_index / resolution) * self.parameters['radius']
                
            bbox = o3d.geometry.AxisAlignedBoundingBox()
            bbox.min_bound = [-2 * self.parameters['radius']] * 3
            bbox.max_bound = [2 * self.parameters['radius']] * 2 + [nearest_height]

            self.mesh = self.mesh.crop(bbox)
            boundary_edges = np.asarray(self.mesh.get_non_manifold_edges(allow_boundary_edges = False))
            boundary_vertices = np.reshape(boundary_edges, -1)
            boundary_vertices = np.unique(boundary_vertices)

            vertices = np.asarray(self.mesh.vertices)
            nearest_elevation = math.pi * nearset_elevation_index / resolution
            elevation = math.acos((self.parameters['radius'] - self.parameters['height']) / self.parameters['radius'])
            for boundary_vertex in boundary_vertices:
                vertices[boundary_vertex, 0:2] = vertices[boundary_vertex, 0:2] * math.sin(elevation) / math.sin(nearest_elevation)
                vertices[boundary_vertex, 2] = self.parameters['radius'] * math.cos(elevation)

            center_vertex = [0, 0, self.parameters['radius'] * math.cos(elevation)]
            vertices = np.append(vertices, [center_vertex], axis = 0)
            center_ver_ind = vertices.shape[0] - 1

            triangles = np.asarray(self.mesh.triangles)
            plane_triangles = np.concatenate((boundary_edges, np.ones((boundary_edges.shape[0], 1)) * center_ver_ind), axis = 1).astype(int)
            plane_normals = np.cross(vertices[plane_triangles[:, 1]] - vertices[plane_triangles[:, 0]], vertices[plane_triangles[:, 2]] - vertices[plane_triangles[:, 1]])
            plane_triangles = np.where(np.repeat(np.transpose([plane_normals[:, 2]]), 3, axis = 1) > 0, plane_triangles, np.transpose([plane_triangles[:, 1], plane_triangles[:, 0], plane_triangles[:, 2]]))
            triangles = np.append(triangles, plane_triangles, axis = 0)
    
            self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self.mesh.triangles = o3d.utility.Vector3iVector(triangles)

        self.transform_mesh()

class Cone(Base_primitives):
 
    def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8]):
        if parameters['upper_radius'] >= parameters['lower_radius']:
            warnings.warn("Set upper_radius smaller than lower_radius")
        self.type = 'cone'
        self.SE3 = SE3
        self.parameters = parameters
        self.color = color
        if parameters['upper_radius'] is 0:
            self.mesh = o3d.geometry.TriangleMesh.create_cone(radius=self.parameters['lower_radius'], height=self.parameters['height'], resolution = 100)
            self.mesh.translate(np.array([0, 0, - self.parameters['height'] / 2]))
        else:
            self.mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=self.parameters['lower_radius'], height=self.parameters['height'], resolution=100, split=1)
            vertices = np.asarray(self.mesh.vertices)
            upper_plane_ver = np.argwhere(vertices[:, 2] > self.parameters['height'] / 2 - 1e-10)
            upper_plane_ver = np.delete(upper_plane_ver, np.argwhere(upper_plane_ver == 0)) # remove center vertex

            vertices[upper_plane_ver, 0:2] = vertices[upper_plane_ver, 0:2] * self.parameters['upper_radius'] / self.parameters['lower_radius']
            self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.transform_mesh()

class Torus(Base_primitives):
    
    def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8], collisionBox=False):
        self.type = 'torus'
        self.SE3 = SE3
        self.parameters = parameters
        self.color = color
        self.mesh = o3d.geometry.TriangleMesh.create_torus(self.parameters['torus_radius'], self.parameters['tube_radius'], 100, 100)

        vertices_numpy = np.asarray(self.mesh.vertices)
        # eta = np.arcsin(vertices_numpy[:,2:3])
        normxy = np.sqrt(vertices_numpy[:,1:2]**2 + vertices_numpy[:,0:1]**2) - self.parameters['torus_radius']
        eta = np.arctan2(vertices_numpy[:,2:3], normxy)
        omega = np.arctan2(vertices_numpy[:,1:2], vertices_numpy[:,0:1])	

        outer_indices = (np.squeeze(np.logical_or(omega < -self.parameters['c1'], omega > self.parameters['c1']))).nonzero()
        self.mesh.remove_vertices_by_index(outer_indices[0])

        boundary_edges = np.asarray(self.mesh.get_non_manifold_edges(allow_boundary_edges = False))
        boundary_vertices = np.reshape(boundary_edges, -1)
        boundary_vertices = np.unique(boundary_vertices)

        vertices = np.asarray(self.mesh.vertices)
        for boundary_vertex in boundary_vertices:
            normxy_b = np.linalg.norm(vertices[boundary_vertex,0:2]) - self.parameters['torus_radius']
            eta_b = np.arctan2(vertices[boundary_vertex, 2], normxy_b)
            vertices[boundary_vertex, 0:2] = [(self.parameters['torus_radius'] + self.parameters['tube_radius'] * np.cos(eta_b)) * np.cos(self.parameters['c1']), np.sign(vertices[boundary_vertex, 1]) * (self.parameters['torus_radius'] + self.parameters['tube_radius'] * np.cos(eta_b)) * np.sin(abs(self.parameters['c1']))]
        
        centers = np.array([[self.parameters['torus_radius'] * np.cos(self.parameters['c1']), self.parameters['torus_radius'] * np.sin(self.parameters['c1']), 0], [self.parameters['torus_radius'] * np.cos(-self.parameters['c1']), self.parameters['torus_radius'] * np.sin(-self.parameters['c1']), 0]])
        vertices = np.concatenate((vertices, centers), axis=0)
        centers_ind = [vertices.shape[0] - 2, vertices.shape[0] - 1]

        triangles = np.asarray(self.mesh.triangles)
        positive_boundary_edges = boundary_edges[np.squeeze(np.argwhere(vertices[boundary_edges[:, 0], 1]>=0))]
        negative_boundary_edges = boundary_edges[np.squeeze(np.argwhere(vertices[boundary_edges[:, 0], 1]<0))]
        boundary_edges = [positive_boundary_edges, negative_boundary_edges]
        for section in range(2):
            plane_triangles = np.concatenate((boundary_edges[section], np.ones((boundary_edges[section].shape[0], 1)) * centers_ind[section]), axis = 1).astype(int)
            plane_normals = np.cross(vertices[plane_triangles[:, 1]] - vertices[plane_triangles[:, 0]], vertices[plane_triangles[:, 2]] - vertices[plane_triangles[:, 1]])
            # plane_triangles = np.where(np.repeat(np.transpose([plane_normals[:, 2]]), 3, axis = 1) > 0, plane_triangles, np.transpose([plane_triangles[:, 1], plane_triangles[:, 0], plane_triangles[:, 2]]))

            opposite_normal_ind = np.squeeze(np.argwhere(plane_normals[:, 0] > 0))
            plane_triangles[opposite_normal_ind] = np.flip(plane_triangles[opposite_normal_ind], -1)

            triangles = np.append(triangles, plane_triangles, axis = 0)			
    
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.triangles = o3d.utility.Vector3iVector(triangles)

        # self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        # self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        
        self.transform_mesh()

# class Torus(Base_primitives):
    
#     def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8], collisionBox=False):
#         self.type = 'torus'
#         self.SE3 = SE3
#         self.parameters = parameters
#         self.color = color
#         self.mesh = o3d.geometry.TriangleMesh.create_torus(self.parameters['torus_radius'], self.parameters['tube_radius'], 100, 100)
#         self.transform_mesh()

# class Torus(Base_primitives):
    
# 	def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8], collisionBox=False):
# 		self.type = 'torus'
# 		self.SE3 = SE3
# 		self.parameters = parameters
# 		self.color = color
# 		self.mesh = o3d.geometry.TriangleMesh.create_torus(self.parameters['torus_radius'], self.parameters['tube_radius'], 100, 100)
		
# 		if parameters['c1'] > - (self.parameters['torus_radius'] + self.parameters['tube_radius']) and parameters['c1'] < (self.parameters['torus_radius'] + self.parameters['tube_radius']):
# 			bbox = o3d.geometry.AxisAlignedBoundingBox()
# 			bbox.min_bound = [- (self.parameters['torus_radius'] + self.parameters['tube_radius'])] * 3
# 			bbox.max_bound = [self.parameters['c1'], (self.parameters['torus_radius'] + self.parameters['tube_radius']), (self.parameters['torus_radius'] + self.parameters['tube_radius'])]
# 			self.mesh = self.mesh.crop(bbox)
# 			boundary_edges = np.asarray(self.mesh.get_non_manifold_edges(allow_boundary_edges = False))
# 			boundary_vertices = np.reshape(boundary_edges, -1)
# 			boundary_vertices = np.unique(boundary_vertices)

# 			# d = 2 * (self.parameters['torus_radius']**2 + self.parameters['tube_radius']**2 - self.parameters['c1']**2)
# 			# e = 2 * (- self.parameters['torus_radius']**2 + self.parameters['tube_radius']**2 - self.parameters['c1']**2)
# 			# f = - (self.parameters['tube_radius'] + self.parameters['torus_radius'] + self.parameters['c1']) * (self.parameters['tube_radius'] + self.parameters['torus_radius'] - self.parameters['c1']) * (self.parameters['tube_radius'] - self.parameters['torus_radius'] + self.parameters['c1']) * (self.parameters['tube_radius'] - self.parameters['torus_radius'] - self.parameters['c1'])
# 			for boundary_vertex in boundary_vertices.tolist():
# 				vertex_pnt = self.mesh.vertices[boundary_vertex]
# 				sinv = vertex_pnt[2] / self.parameters['tube_radius']
# 				cosv = (np.linalg.norm(vertex_pnt[0:2]) - self.parameters['torus_radius']) / self.parameters['tube_radius']
# 				cosu_proj = self.parameters['c1'] / (self.parameters['torus_radius'] + self.parameters['tube_radius'] * cosv)
# 				if cosu_proj > 1.0:
# 					u_proj = 0
# 				else:
# 					u_proj = np.arccos(cosu_proj)
# 				self.mesh.vertices[boundary_vertex] = [self.parameters['c1'], (self.parameters['torus_radius'] + self.parameters['tube_radius'] * cosv) * np.sin(np.sign(vertex_pnt[1]) * u_proj), self.parameters['tube_radius'] * sinv] 
			
# 			vertices = np.asarray(self.mesh.vertices)
# 			# for i, boundary_vertex in enumerate(boundary_vertices.tolist()):
# 			# 	vertex_pnt = self.mesh.vertices[boundary_vertex]
# 			# 	theta = np.arctan2(vertex_pnt[2], vertex_pnt[1])
# 			# 	b = - (d * np.cos(theta)**2 + e * np.sin(theta) **2)
# 			# 	c = - f
# 			# 	r_squared1 = (- b + np.sqrt(b**2 - 4 * c)) / 2
# 			# 	r_squared2 = (- b - np.sqrt(b**2 - 4 * c)) / 2
# 			# 	if b**2 - 4*c < 0:
# 			# 		if theta < np.pi / 2 and theta >= 0:
# 			# 			theta = np.arccos(np.sqrt((np.sqrt(-4 * f) - e) / (d - e)))
# 			# 		elif theta >= - np.pi / 2 and theta < 0:
# 			# 			theta = - np.arccos(np.sqrt((np.sqrt(-4 * f) - e) / (d - e)))
# 			# 		elif theta <= np. pi and theta >= np. pi / 2 :
# 			# 			theta = np.arccos( - np.sqrt((np.sqrt(-4 * f) - e) / (d - e)))
# 			# 		else:
# 			# 			theta = - np.arccos( - np.sqrt((np.sqrt(-4 * f) - e) / (d - e)))
# 			# 		b = - (d * (np.cos(theta))**2 + e * (np.sin(theta)) **2)
# 			# 		r_squared1 = (- b ) / 2
# 			# 		r_squared2 = (- b ) / 2
				
# 			# 	if r_squared2 > 0 and r_squared1 != r_squared2:
# 			# 		r1 = np.sqrt(r_squared1)
# 			# 		r2 = np.sqrt(r_squared2)
# 			# 		r_original = np.linalg.norm(vertex_pnt[1:])
# 			# 		r_cand = [r1, r2]
# 			# 		r = r_cand[np.argmin(abs(np.array(r_cand) - r_original))]
# 			# 	else:
# 			# 		r = np.sqrt(r_squared1)
# 			# 	vertices[boundary_vertex] = [self.parameters['c1'], r * np.cos(theta), r * np.sin(theta)]

# 			if self.parameters['c1'] > self.parameters['torus_radius'] - self.parameters['tube_radius'] or self.parameters['c1'] < -self.parameters['torus_radius'] + self.parameters['tube_radius']:
# 				vertices = np.concatenate((vertices, np.array([[self.parameters['c1'], 0, 0]])), axis=0)
# 				center_ver_ind = vertices.shape[0] - 1

# 				triangles = np.asarray(self.mesh.triangles)
# 				plane_triangles = np.concatenate((boundary_edges, np.ones((boundary_edges.shape[0], 1)) * center_ver_ind), axis = 1).astype(int)
# 				plane_normals = np.cross(vertices[plane_triangles[:, 1]] - vertices[plane_triangles[:, 0]], vertices[plane_triangles[:, 2]] - vertices[plane_triangles[:, 1]])
# 				plane_triangles = np.where(np.repeat(np.transpose([plane_normals[:, 2]]), 3, axis = 1) > 0, plane_triangles, np.transpose([plane_triangles[:, 1], plane_triangles[:, 0], plane_triangles[:, 2]]))

# 				opposite_normal_ind = np.squeeze(np.argwhere(plane_normals[:, 0] > 0))
# 				plane_triangles[opposite_normal_ind] = np.flip(plane_triangles[opposite_normal_ind], 1)

# 				triangles = np.append(triangles, plane_triangles, axis = 0)
# 			else:
# 				centers = np.array([[self.parameters['c1'], np.sqrt(self.parameters['torus_radius']**2 - self.parameters['c1']**2), 0], [self.parameters['c1'], -np.sqrt(self.parameters['torus_radius']**2 - self.parameters['c1']**2), 0]])
# 				vertices = np.concatenate((vertices, centers), axis=0)
# 				centers_ind = [vertices.shape[0] - 2, vertices.shape[0] - 1]

# 				triangles = np.asarray(self.mesh.triangles)
# 				positive_boundary_edges = boundary_edges[np.squeeze(np.argwhere(vertices[boundary_edges[:, 0], 1]>=0))]
# 				negative_boundary_edges = boundary_edges[np.squeeze(np.argwhere(vertices[boundary_edges[:, 0], 1]<0))]
# 				boundary_edges = [positive_boundary_edges, negative_boundary_edges]
# 				for section in range(2):
# 					plane_triangles = np.concatenate((boundary_edges[section], np.ones((boundary_edges[section].shape[0], 1)) * centers_ind[section]), axis = 1).astype(int)
# 					plane_normals = np.cross(vertices[plane_triangles[:, 1]] - vertices[plane_triangles[:, 0]], vertices[plane_triangles[:, 2]] - vertices[plane_triangles[:, 1]])
# 					plane_triangles = np.where(np.repeat(np.transpose([plane_normals[:, 2]]), 3, axis = 1) > 0, plane_triangles, np.transpose([plane_triangles[:, 1], plane_triangles[:, 0], plane_triangles[:, 2]]))

# 					opposite_normal_ind = np.squeeze(np.argwhere(plane_normals[:, 0] > 0))
# 					plane_triangles[opposite_normal_ind] = np.flip(plane_triangles[opposite_normal_ind], -1)

# 					triangles = np.append(triangles, plane_triangles, axis = 0)			
	
# 			self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
# 			self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
		
		
# 		self.transform_mesh()

# class RectangleRing(Base_primitives):

# 	def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8], collisionBox=False):
# 		self.type = 'rectangle_ring'
# 		self.SE3 = SE3
# 		self.parameters = parameters
# 		self.color = color

# 		P1_SE3 = lie_alg.define_SE3(np.identity(3), [parameters['width'] / 2 - parameters['thickness'] / 2, 0, 0])
# 		P2_SE3 = lie_alg.define_SE3(np.identity(3), [- parameters['width'] / 2 + parameters['thickness'] / 2, 0, 0])
# 		P3_SE3 = lie_alg.define_SE3(np.identity(3), [0, parameters['depth'] / 2 - parameters['thickness'] / 2, 0])
# 		P4_SE3 = lie_alg.define_SE3(np.identity(3), [0, - parameters['depth'] / 2 + parameters['thickness'] / 2, 0])

# 		P1_parameters = {"width": parameters['thickness'], "depth": parameters['depth'], "height": parameters['height']}
# 		P2_parameters = {"width": parameters['thickness'], "depth": parameters['depth'], "height": parameters['height']}
# 		P3_parameters = {"width": parameters['width'] - 2 * parameters['thickness'] , "depth": parameters['thickness'], "height": parameters['height']}
# 		P4_parameters = {"width": parameters['width'] - 2 * parameters['thickness'], "depth": parameters['thickness'], "height": parameters['height']}

# 		P1 = Box(SE3=P1_SE3, parameters=P1_parameters, color=self.color)
# 		P2 = Box(SE3=P2_SE3, parameters=P2_parameters, color=self.color)
# 		P3 = Box(SE3=P3_SE3, parameters=P3_parameters, color=self.color)
# 		P4 = Box(SE3=P4_SE3, parameters=P4_parameters, color=self.color)

# 		self.mesh = P1.mesh + P2.mesh + P3.mesh + P4.mesh

# 		self.transform_mesh()


class RectangleRing(Base_primitives):

	def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8], collisionBox=False):
		self.type = 'rectangle_ring'
		self.SE3 = SE3
		self.parameters = parameters
		self.color = color

		supertoroid_parameters = {'a1': (self.parameters['width'] / 2 - self.parameters['thickness1']) / (self.parameters['depth'] / self.parameters['thickness2'] - 2)  , 'a2': self.parameters['thickness2'] / 2 , 'a3': self.parameters['height'] / 2 , 'a4': (self.parameters['depth'] / 2 - self.parameters['thickness2'] / 2) * 2 / self.parameters['thickness2'], 'e1': 0, 'e2': 0}
		rectangle_ring = Supertoroid(SE3=np.identity(4), parameters=supertoroid_parameters, color=color)
		self.mesh = rectangle_ring.mesh
		self.mesh.remove_duplicated_vertices()
		self.mesh.remove_degenerate_triangles()
		vertices = np.asarray(self.mesh.vertices)
		x_top_vertices = np.squeeze(np.argwhere(vertices[:,0] == max(vertices[:, 0])))
		x_bottom_vertices = np.squeeze(np.argwhere(vertices[:,0] == min(vertices[:, 0])))
		vertices[x_top_vertices, 0] = self.parameters['width'] / 2
		vertices[x_bottom_vertices, 0] = - self.parameters['width'] / 2
		self.mesh.vertices = o3d.utility.Vector3dVector(vertices)

		if 'c1' in self.parameters.keys() and self.parameters['c1'] < self.parameters['width'] / 2 and self.parameters['c1'] > self.parameters['width'] / 2 - self.parameters['thickness1']:
			vertices[x_top_vertices, 0] = self.parameters['c1']
			self.mesh.vertices = o3d.utility.Vector3dVector(vertices)

		elif 'c1' in self.parameters.keys() and self.parameters['c1'] <= self.parameters['width'] / 2 - self.parameters['thickness1'] and self.parameters['c1'] > - self.parameters['width'] / 2 + self.parameters['thickness1']:
			self.mesh = self.mesh.subdivide_midpoint()
			bbox = o3d.geometry.AxisAlignedBoundingBox()
			bbox.min_bound = [-self.parameters['width'] / 2, - self.parameters['depth'] / 2, - self.parameters['height'] / 2]
			bbox.max_bound = [self.parameters['width'] / 2 - self.parameters['thickness1'] - 0.001, self.parameters['depth'] / 2, self.parameters['height'] / 2]
			self.mesh =self.mesh.crop(bbox)

			boundary_edges = np.asarray(self.mesh.get_non_manifold_edges(allow_boundary_edges = False))
			boundary_vertices = np.reshape(boundary_edges, -1)
			boundary_vertices = np.unique(boundary_vertices)

			vertices = np.asarray(self.mesh.vertices)
			vertices[boundary_vertices, 0] = self.parameters['c1']
			self.mesh.vertices = o3d.utility.Vector3dVector(vertices)


			triangles = np.asarray(self.mesh.triangles)
			triangles_sliced_surface1 = np.array([[11, 18, 12], [11, 15, 18], [11, 13, 15], [15, 16, 18], [12, 18, 24], [18, 21, 24], [18, 19, 21], [21, 22, 24]])
			triangles_sliced_surface2 = np.array([[40, 41, 45], [40, 45, 43], [40, 43, 42], [43, 45, 44], [41, 49, 45], [45, 47, 46], [45, 49, 47], [47, 49, 48]])

			triangles = np.concatenate((triangles, triangles_sliced_surface1, triangles_sliced_surface2), axis = 0)
			self.mesh.triangles = o3d.utility.Vector3iVector(triangles)

		self.transform_mesh()


class CylinderRing(Base_primitives):

	def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8], resolution=100, collisionBox=False):
		self.type = 'cylinder_ring'
		self.SE3 = SE3
		self.parameters = parameters
		self.color = color

		radiuses = [self.parameters['radius_outer'], self.parameters['radius_inner']]
		meshes = [None] * 2
		for i in range(2):
			mesh = o3d.geometry.TriangleMesh.create_cylinder(radius = radiuses[i], height = self.parameters['height'], resolution = resolution)
			mesh.remove_vertices_by_index([0, 1])
			meshes[i] = mesh

		meshes[1].triangles = o3d.utility.Vector3iVector(np.flip(np.asarray(meshes[1].triangles), 1))
		self.mesh = meshes[0] + meshes[1]
		boundary_edges = np.asarray(self.mesh.get_non_manifold_edges(allow_boundary_edges = False))
		boundary_vertices = np.reshape(boundary_edges, -1)
		boundary_vertices = np.unique(boundary_vertices)
		boundary_vertices1 = boundary_vertices[:len(boundary_vertices) // 4]
		boundary_vertices2 = boundary_vertices[len(boundary_vertices) // 4:len(boundary_vertices) // 2]
		num_ver_outer = len(meshes[0].vertices)

		triangles = np.asarray(self.mesh.triangles)
		triangles_top1 = np.concatenate((np.expand_dims(boundary_vertices1, axis=1), np.expand_dims(boundary_vertices1[list(range(1,len(boundary_vertices1)))+[0]], axis=1), np.expand_dims(boundary_vertices1 + num_ver_outer, axis=1)), axis = 1)
		triangles_top2 = np.concatenate((np.expand_dims(boundary_vertices1[list(range(1,len(boundary_vertices1)))+[0]], axis=1), np.expand_dims(boundary_vertices1[list(range(1,len(boundary_vertices1)))+[0]] + num_ver_outer, axis=1), np.expand_dims(boundary_vertices1 + num_ver_outer, axis=1)), axis = 1)
		triangles_bottom1 = np.concatenate((np.expand_dims(boundary_vertices2[list(range(1,len(boundary_vertices2)))+[0]], axis=1), np.expand_dims(boundary_vertices2, axis=1), np.expand_dims(boundary_vertices2 + num_ver_outer, axis=1)), axis = 1)
		triangles_bottom2 = np.concatenate((np.expand_dims(boundary_vertices2[list(range(1,len(boundary_vertices2)))+[0]] + num_ver_outer, axis=1), np.expand_dims(boundary_vertices2[list(range(1,len(boundary_vertices2)))+[0]], axis=1), np.expand_dims(boundary_vertices2 + num_ver_outer, axis=1)), axis = 1)

		triangles = np.concatenate((triangles, triangles_top1, triangles_top2, triangles_bottom1, triangles_bottom2), axis = 0) 

		self.mesh.triangles = o3d.utility.Vector3iVector(triangles)

		if 'c1' in self.parameters.keys() and self.parameters['c1'] < self.parameters['radius_inner'] and self.parameters['c1'] > -self.parameters['radius_inner']:
			bbox = o3d.geometry.AxisAlignedBoundingBox()
			bbox.min_bound = [-self.parameters['radius_outer'], -self.parameters['radius_outer'], - self.parameters['height'] / 2]
			bbox.max_bound = [self.parameters['c1'], self.parameters['radius_outer'], self.parameters['height'] / 2]

			self.mesh = self.mesh.crop(bbox)
			# top and bottom plane
			boundary_edges = np.asarray(self.mesh.get_non_manifold_edges(allow_boundary_edges = False))
			boundary_vertices = np.reshape(boundary_edges, -1)
			boundary_vertices = np.unique(boundary_vertices)

			triangles = np.asarray(self.mesh.triangles)
			vertices = np.asarray(self.mesh.vertices)
			top_vertices = boundary_vertices[np.squeeze(np.argwhere(vertices[boundary_vertices, 2] == self.parameters['height'] / 2))]
			bottom_vertices = boundary_vertices[np.squeeze(np.argwhere(vertices[boundary_vertices, 2] == - self.parameters['height'] / 2))]
			top_bottom_vertices = [top_vertices, bottom_vertices]
			plane_normals = np.array([[0, 0, 1], [0, 0, -1]])
			for i in range(2):
				if len(top_bottom_vertices[i]) > 4:
					y_pos_vertices = top_bottom_vertices[i][np.squeeze(np.argwhere(vertices[top_bottom_vertices[i], 1] > 0))]
					y_neg_vertices = top_bottom_vertices[i][np.squeeze(np.argwhere(vertices[top_bottom_vertices[i], 1] < 0))]
					all_vertices = [y_pos_vertices, y_neg_vertices]
					for j in range(2):
						if len(all_vertices[j]) > 2 and self.parameters['c1'] >=0:
							if i == 0:
								plane_triangles =  np.concatenate((np.expand_dims(all_vertices[j][2:], axis=1), np.expand_dims(all_vertices[j][1:-1], axis=1), all_vertices[j][0] * np.ones((len(all_vertices[j]) - 2, 1))), axis=1)
							else:
								plane_triangles =  np.concatenate((np.expand_dims(all_vertices[j][1:-1], axis=1), np.expand_dims(all_vertices[j][2:], axis=1), all_vertices[j][0] * np.ones((len(all_vertices[j]) - 2, 1))), axis=1)
							triangles = np.append(triangles, plane_triangles, axis=0)
						elif  len(all_vertices[j]) > 2 and self.parameters['c1'] <0:
							if i == 0:
								plane_triangles =  np.concatenate((np.expand_dims(all_vertices[j][:-2], axis=1), np.expand_dims(all_vertices[j][1:-1], axis=1), all_vertices[j][-1] * np.ones((len(all_vertices[j]) - 2, 1))), axis=1)
							else:
								plane_triangles =  np.concatenate((np.expand_dims(all_vertices[j][1:-1], axis=1), np.expand_dims(all_vertices[j][:-2], axis=1), all_vertices[j][-1] * np.ones((len(all_vertices[j]) - 2, 1))), axis=1)
							triangles = np.append(triangles, plane_triangles, axis=0)

			self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
			self.mesh.triangles = o3d.utility.Vector3iVector(triangles)

			# sliced plane
			boundary_edges = np.asarray(self.mesh.get_non_manifold_edges(allow_boundary_edges = False))
			boundary_vertices = np.reshape(boundary_edges, -1)
			boundary_vertices = np.unique(boundary_vertices)

			outer_boundary_vertices = boundary_vertices[:len(boundary_vertices) // 2]
			inner_boundary_vertices = boundary_vertices[len(boundary_vertices) // 2:]
			outer_boundary_vertices_above = outer_boundary_vertices[::2]
			outer_boundary_vertices_below = outer_boundary_vertices[1::2]
			inner_boundary_vertices_above = inner_boundary_vertices[::2]
			inner_boundary_vertices_below = inner_boundary_vertices[1::2]

			sliced_plane_triangles1 = np.concatenate((np.expand_dims(outer_boundary_vertices_above[1:], axis=1), np.expand_dims(outer_boundary_vertices_above[:-1], axis=1), np.expand_dims(inner_boundary_vertices_above[1:], axis=1)), axis=1)
			sliced_plane_triangles2 = np.concatenate((np.expand_dims(outer_boundary_vertices_above[:-1], axis=1), np.expand_dims(inner_boundary_vertices_above[:-1], axis=1), np.expand_dims(inner_boundary_vertices_above[1:], axis=1)), axis=1)
			sliced_plane_triangles3 = np.concatenate((np.expand_dims(outer_boundary_vertices_below[:-1], axis=1), np.expand_dims(outer_boundary_vertices_below[1:], axis=1), np.expand_dims(inner_boundary_vertices_below[1:], axis=1)), axis=1)
			sliced_plane_triangles4 = np.concatenate((np.expand_dims(inner_boundary_vertices_below[:-1], axis=1), np.expand_dims(outer_boundary_vertices_below[:-1], axis=1), np.expand_dims(inner_boundary_vertices_below[1:], axis=1)), axis=1)

			triangles = np.concatenate((triangles, sliced_plane_triangles1, sliced_plane_triangles2, sliced_plane_triangles3, sliced_plane_triangles4), axis = 0) 

			vertices[outer_boundary_vertices, :2] = np.concatenate((self.parameters['c1'] * np.ones((len(outer_boundary_vertices),1)), np.expand_dims(np.sign(vertices[outer_boundary_vertices, 1]) * np.sqrt(parameters['radius_outer']**2 - self.parameters['c1']**2), axis=1)), axis=1)
			vertices[inner_boundary_vertices, :2] = np.concatenate((self.parameters['c1'] * np.ones((len(inner_boundary_vertices),1)), np.expand_dims(np.sign(vertices[inner_boundary_vertices, 1]) * np.sqrt(parameters['radius_inner']**2 - self.parameters['c1']**2), axis=1)), axis=1)

			self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
			self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
		self.transform_mesh()

class Semi_Sphere_Shell(Base_primitives):

	def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8], resolution=100, collisionBox=False):
		self.type = 'semi_sphere_shell'
		self.SE3 = SE3
		self.parameters = parameters
		self.color = color
		
		bbox = o3d.geometry.AxisAlignedBoundingBox()
		bbox.min_bound = [-2 * self.parameters['radius_outer']] * 3
		bbox.max_bound = [2 * self.parameters['radius_outer']] * 2 + [0]

		radiuses = [self.parameters['radius_outer'], self.parameters['radius_inner']]
		meshes = [None] * 2
		for i, mesh in enumerate(meshes):
			mesh = o3d.geometry.TriangleMesh.create_sphere(radius = radiuses[i], resolution = resolution)
			mesh = mesh.crop(bbox)

			boundary_edges = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges = False))
			boundary_vertices = np.reshape(boundary_edges, -1)
			boundary_vertices = np.unique(boundary_vertices)

			elevation = np.pi / 2
			for boundary_vertex in boundary_vertices:
				xy_norm = np.linalg.norm(mesh.vertices[boundary_vertex][0:2])
				mesh.vertices[boundary_vertex] = [radiuses[i] * mesh.vertices[boundary_vertex][0] / xy_norm, radiuses[i] * mesh.vertices[boundary_vertex][1] / xy_norm, 0]
			meshes[i] = mesh

		meshes[1].triangles = o3d.utility.Vector3iVector(np.flip(np.asarray(meshes[1].triangles), 1))
		self.mesh = meshes[0] + meshes[1]
		num_ver_outer = np.asarray(meshes[0].vertices).shape[0]

		triangles = np.asarray(self.mesh.triangles)
		triangles_top1 = np.concatenate((np.expand_dims(boundary_vertices, axis=1), np.expand_dims(boundary_vertices[list(range(1,len(boundary_vertices)))+[0]], axis=1), np.expand_dims(boundary_vertices + num_ver_outer, axis=1)), axis = 1)
		triangles_top2 = np.concatenate((np.expand_dims(boundary_vertices[list(range(1,len(boundary_vertices)))+[0]], axis=1), np.expand_dims(boundary_vertices[list(range(1,len(boundary_vertices)))+[0]] + num_ver_outer, axis=1), np.expand_dims(boundary_vertices + num_ver_outer, axis=1)), axis = 1)
		triangles = np.concatenate((triangles, triangles_top1, triangles_top2), axis = 0) 
		self.mesh.triangles = o3d.utility.Vector3iVector(triangles)

		self.transform_mesh()

class Superquadric(Base_primitives):
    
    def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8]):
        
        self.type = 'superquadric'
        self.SE3 = SE3
        self.parameters = parameters
        self.color = color
        self.n_samples = 50

        # make pointcloud
        # self.pcd = o3d.geometry.PointCloud()
        # self.points_numpy = vertices_superquadric(self.parameters, self.SE3, n_samples = self.n_samples)
        # self.pcd.points = o3d.utility.Vector3dVector(self.points_numpy.transpose())
        # self.pcd.estimate_normals()

        # # estimate radius for rolling ball
        # distances = self.pcd.compute_nearest_neighbor_distance()
        # avg_dist = np.mean(distances)
        # radius = 1.5 * avg_dist  

        # make mesh

        self.mesh = mesh_superquadric(self.parameters, self.SE3, n_samples=self.n_samples)

        # self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        #     self.pcd,
        #     o3d.utility.DoubleVector([radius, radius * 2]))
        # self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        #     self.pcd,
        #     o3d.utility.DoubleVector([0.005, 0.01, 0.02, 0.04]))
        # self.mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        #     self.pcd, 
        #     depth=9)

        # self.V = vertices_superquadric(self.parameters, self.SE3, n_samples = self.n_samples)
        # self.M = trimesh.Trimesh(vertices = self.V.transpose()).convex_hull

        # self.mesh = o3d.geometry.TriangleMesh()
        # self.mesh.vertices = o3d.utility.Vector3dVector()
        # self.mesh.faces = 0

        self.transform_mesh()

class Extended_Superquadric(Base_primitives):
    
    def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8]):
        
        self.type = 'extended_superquadric'
        self.SE3 = SE3
        self.parameters = parameters
        self.color = color
        self.n_samples = 50

        # make mesh
        self.mesh = mesh_extended_superquadric(self.parameters, self.SE3, n_samples=self.n_samples)

        self.transform_mesh()


class Supertoroid(Base_primitives):
    
    def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8]):
        
        self.type = 'supertoroid'
        self.SE3 = SE3
        self.parameters = parameters
        self.color = color
        self.n_samples = 50

        # make mesh
        self.mesh = mesh_supertoroid(self.parameters, self.SE3, n_samples=self.n_samples)

        self.transform_mesh()

class Extended_Supertoroid(Base_primitives):
    
    def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8]):
        
        self.type = 'extended_supertoroid'
        self.SE3 = SE3
        self.parameters = parameters
        self.color = color
        self.n_samples = 50

        # make mesh
        self.mesh = mesh_extended_supertoroid(self.parameters, self.SE3, n_samples=self.n_samples)

        self.transform_mesh()

class Deformed_Superquadric(Base_primitives):
    
    def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8]):
        
        self.type = 'deformed_superquadric'
        self.SE3 = SE3
        self.parameters = parameters
        self.color = color
        self.n_samples = 50

        # make mesh
        self.mesh = mesh_deformed_superquadric(self.parameters, self.SE3, n_samples=self.n_samples)

        self.transform_mesh()

def vertices_superquadric(parameters, SE3, n_samples=100):

    assert SE3.shape == (4, 4)

    # parameters
    a1 = parameters['a1']
    a2 = parameters['a2']
    a3 = parameters['a3']
    e1 = parameters['e1']
    e2 = parameters['e2']
    R = SE3[0:3, 0:3]
    t = SE3[0:3, 3:]

    # make grids
    eta = np.linspace(-np.pi/2, np.pi/2, n_samples, endpoint=True)
    omega = np.linspace(-np.pi, np.pi, n_samples, endpoint=True)
    eta, omega = np.meshgrid(eta, omega)

    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)

    # Get an array of size 3x10000 that contains the points of the SQ
    points = np.stack([x, y, z]).reshape(3, -1)
    points_transformed = R.transpose().dot(points) + t

    # print(points_transformed.shape)
    return points_transformed

def mesh_superquadric(parameters, SE3, n_samples=100):

    assert SE3.shape == (4, 4)

    # parameters
    a1 = parameters['a1']
    a2 = parameters['a2']
    a3 = parameters['a3']
    e1 = parameters['e1']
    e2 = parameters['e2']
    R = SE3[0:3, 0:3]
    t = SE3[0:3, 3:]

    # make grids
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius = 1, resolution = 30)
    vertices_numpy = np.asarray(mesh.vertices)
    eta = np.arcsin(vertices_numpy[:,2:3])
    omega = np.arctan2(vertices_numpy[:,1:2], vertices_numpy[:,0:1])

    # make new vertices
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)

    # reconstruct point matrix
    points = np.concatenate((x, y, z), axis=1)
    # points = np.stack([x, y, z])
    # points_transformed = R.transpose().dot(points) + t

    mesh.vertices = o3d.utility.Vector3dVector(points)

    return mesh

# def mesh_deformed_superquadric(parameters, SE3, n_samples=100):
    
#     assert SE3.shape == (4, 4)

#     # parameters
#     a1 = parameters['a1']
#     a2 = parameters['a2']
#     a3 = parameters['a3']
#     e1 = parameters['e1']
#     e2 = parameters['e2']
#     # kx = parameters['kx']
#     # ky = parameters['ky']
#     k = parameters['k']
#     R = SE3[0:3, 0:3]
#     t = SE3[0:3, 3:]

#     # make grids
#     mesh = o3d.geometry.TriangleMesh.create_sphere(radius = 1, resolution = 30)
#     vertices_numpy = np.asarray(mesh.vertices)
#     eta = np.arcsin(vertices_numpy[:,2:3])
#     omega = np.arctan2(vertices_numpy[:,1:2], vertices_numpy[:,0:1])

#     # make new vertices
#     x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
#     y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
#     z = a3 * fexp(np.sin(eta), e1)

#     # tampering
#     f_x = k / a3 * z + 1
#     f_y = k / a3 * z + 1
#     x = f_x * x
#     y = f_y * y

#     # reconstruct point matrix
#     points = np.concatenate((x, y, z), axis=1)
#     # points = np.stack([x, y, z])
#     # points_transformed = R.transpose().dot(points) + t

#     mesh.vertices = o3d.utility.Vector3dVector(points)

#     return mesh

def mesh_deformed_superquadric(parameters, SE3, n_samples=100):
    
    assert SE3.shape == (4, 4)

    # parameters
    a1 = parameters['a1']
    a2 = parameters['a2']
    a3 = parameters['a3']
    e1 = parameters['e1']
    e2 = parameters['e2']
    # kx = parameters['kx']
    # ky = parameters['ky']
    if 'k' in parameters.keys():
        k = parameters['k']
    if 'b' in parameters.keys():
        # b = parameters['b']
        # b = parameters['b'] / a1
        b = parameters['b'] / np.maximum(a1, a2)
        cos_alpha = parameters['cos_alpha']
        sin_alpha = parameters['sin_alpha']
        alpha = np.arctan2(sin_alpha, cos_alpha)

    # make grids
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius = 1, resolution = 30)
    vertices_numpy = np.asarray(mesh.vertices)
    eta = np.arcsin(vertices_numpy[:,2:3])
    omega = np.arctan2(vertices_numpy[:,1:2], vertices_numpy[:,0:1])

    # make new vertices
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)

    points = np.concatenate((x, y, z), axis=1)
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh = mesh.subdivide_midpoint(2)

    points = np.asarray(mesh.vertices)
    x = points[:, 0:1]
    y = points[:, 1:2]
    z = points[:, 2:3]

    # tampering
    if 'k' in parameters.keys():
        f_x = k / a3 * z + 1
        f_y = k / a3 * z + 1
        x = f_x * x
        y = f_y * y

    # bending
    if 'b' in parameters.keys():
        gamma = z * b
        r = np.cos(alpha - np.arctan2(y, x)) * np.sqrt(x ** 2 + y ** 2)
        R = 1 / b - np.cos(gamma) * (1 / b - r)
        x = x + np.cos(alpha) * (R - r)
        y = y + np.sin(alpha) * (R - r)
        z = np.sin(gamma) * (1 / b - r)

    # reconstruct point matrix
    points = np.concatenate((x, y, z), axis=1)
    # points = np.stack([x, y, z])
    # points_transformed = R.transpose().dot(points) + t

    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()

    return mesh

def mesh_extended_superquadric(parameters, SE3, n_samples=100):
    
    assert SE3.shape == (4, 4)

    # parameters
    a1 = parameters['a1']
    a2 = parameters['a2']
    a3 = parameters['a3']
    e1 = parameters['e1']
    e2 = parameters['e2']
    c1 = parameters['c1']
    c2 = parameters['c2']
    R = SE3[0:3, 0:3]
    t = SE3[0:3, 3:]

    # make grids
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius = 1, resolution = 30)
    vertices_numpy = np.asarray(mesh.vertices)
    eta = np.arcsin(vertices_numpy[:,2:3])
    omega = np.arctan2(vertices_numpy[:,1:2], vertices_numpy[:,0:1])

    # make new vertices
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)

    # # # # boundary   
    x_null = copy.deepcopy(x)
    y_null = copy.deepcopy(y)
    z_null = copy.deepcopy(z)
    x[np.logical_and(z_null <= c1, np.abs(x_null/a1)**(2/e2) + np.abs(y_null/a2)**(2/e2) >= (1 - np.abs(c1/a3)**(2/e1))**(e1/e2))] = 0
    y[np.logical_and(z_null <= c1, np.abs(x_null/a1)**(2/e2) + np.abs(y_null/a2)**(2/e2) >= (1 - np.abs(c1/a3)**(2/e1))**(e1/e2))] = 0
    x[np.logical_and(z_null >= c2, np.abs(x_null/a1)**(2/e2) + np.abs(y_null/a2)**(2/e2) >= (1 - np.abs(c2/a3)**(2/e1))**(e1/e2))] = 0
    y[np.logical_and(z_null >= c2, np.abs(x_null/a1)**(2/e2) + np.abs(y_null/a2)**(2/e2) >= (1 - np.abs(c2/a3)**(2/e1))**(e1/e2))] = 0
    if z_null[z_null > c1].tolist():
        z[z_null <= c1] = np.min(z_null[z_null > c1]) 
    if z_null[z_null < c2].tolist():
        z[z_null >= c2] = np.max(z_null[z_null < c2])
    # if c1 < 0:
        # z[z_null < a3*c1] = a3*c1
    # z[z_null <= c1] = c1 
    # else:
        # z[z_null < a3*c1] = np.min(z_null[z_null > a3*c1])
        # z[z_null < c1] = np.min(z_null[z_null > c1]) 
    # z[z_null > a3*c2] = a3*c2
    # z[z_null >= c2] = c2

    # reconstruct point matrix
    points = np.concatenate((x, y, z), axis=1)
    # points = np.stack([x, y, z])
    # points_transformed = R.transpose().dot(points) + t

    mesh.vertices = o3d.utility.Vector3dVector(points)

    return mesh

def mesh_supertoroid(parameters, SE3, n_samples=100):
    
    assert SE3.shape == (4, 4)

    # parameters
    a1 = parameters['a1']
    a2 = parameters['a2']
    a3 = parameters['a3']
    a4 = parameters['a4']
    e1 = parameters['e1']
    e2 = parameters['e2']
    R = SE3[0:3, 0:3]
    t = SE3[0:3, 3:]

    # make grids
    mesh = o3d.geometry.TriangleMesh.create_torus(torus_radius = 1, 
                                                tube_radius = 0.5, 
                                                radial_resolution = 50,
                                                tubular_resolution = 50
    )
    vertices_numpy = np.asarray(mesh.vertices)
    # eta = np.arcsin(vertices_numpy[:,2:3])
    normxy = np.sqrt(vertices_numpy[:,1:2]**2 + vertices_numpy[:,0:1]**2) - 1
    eta = np.arctan2(vertices_numpy[:,2:3], normxy)
    omega = np.arctan2(vertices_numpy[:,1:2], vertices_numpy[:,0:1])

    # make new vertices
    # x = a1 * (a4 + fexp(np.cos(eta), e1)) * fexp(np.cos(omega), e2)
    # y = a2 * (a4 + fexp(np.cos(eta), e1)) * fexp(np.sin(omega), e2)
    # z = a3 * fexp(np.sin(eta), e1)

    x = a1 * (a4 + fexp(np.cos(eta), e1)) * fexp(np.cos(omega), e2)
    y = a2 * (a4 + fexp(np.cos(eta), e1)) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)

    # reconstruct point matrix
    points = np.concatenate((x, y, z), axis=1)
    # points = np.stack([x, y, z])
    # points_transformed = R.transpose().dot(points) + t

    mesh.vertices = o3d.utility.Vector3dVector(points)

    return mesh

def mesh_extended_supertoroid(parameters, SE3, n_samples=100):
    
    assert SE3.shape == (4, 4)

    # parameters
    a1 = parameters['a1']
    a2 = parameters['a2']
    a3 = parameters['a3']
    a4 = parameters['a4']
    e1 = parameters['e1']
    e2 = parameters['e2']
    c1 = parameters['c1']
    R = SE3[0:3, 0:3]
    t = SE3[0:3, 3:]

    # make grids
    mesh = o3d.geometry.TriangleMesh.create_torus(torus_radius = 1, 
                                                tube_radius = 0.5, 
                                                radial_resolution = 50,
                                                tubular_resolution = 50
    )
    vertices_numpy = np.asarray(mesh.vertices)
    # eta = np.arcsin(vertices_numpy[:,2:3])
    normxy = np.sqrt(vertices_numpy[:,1:2]**2 + vertices_numpy[:,0:1]**2) - 1
    eta = np.arctan2(vertices_numpy[:,2:3], normxy)
    omega = np.arctan2(vertices_numpy[:,1:2], vertices_numpy[:,0:1])

    x = a1 * (a4 + fexp(np.cos(eta), e1)) * fexp(np.cos(omega), e2)
    y = a2 * (a4 + fexp(np.cos(eta), e1)) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)

    # boundary   
    x_null = copy.deepcopy(x)
    y_null = copy.deepcopy(y)
    z_null = copy.deepcopy(z)
    x[x_null >= c1] = np.max(x_null[x_null < c1])
    # y[np.logical_and(x_null >= c1, np.abs(x_null/a1)**(2/e2) + np.abs(y_null/a2)**(2/e2) >= (1 - np.abs(c1/a3)**(2/e1))**(e1/e2))] = 0
    # z[np.logical_and(x_null >= c1, np.abs(x_null/a1)**(2/e2) + np.abs(y_null/a2)**(2/e2) >= (1 - np.abs(c1/a3)**(2/e1))**(e1/e2))] = 0
    
    # z[z_null <= c1] = np.min(z_null[z_null > c1]) 

    # reconstruct point matrix
    points = np.concatenate((x, y, z), axis=1)

    mesh.vertices = o3d.utility.Vector3dVector(points)

    return mesh

def fexp(x, p):
    return np.sign(x)*(np.abs(x)**p)

gen_primitive = {
    "box": Box,
    "cone": Cone,
    "cylinder": Cylinder,
    "sphere": Sphere,
    "torus": Torus,
    "superquadric": Superquadric,
    "extended_superquadric": Extended_Superquadric,
    # "bounded_superquadric": Bounded_Superquadric
    "supertoroid": Supertoroid,
    "extend_supertoroid": Extended_Supertoroid
}

def SO3_from_zaxis(z):
    
	x = np.random.randn(3)  # take a random vector
	x -= x.dot(z) * z       # make it orthogonal to k
	x /= np.linalg.norm(x)  # normalize it
	y = np.cross(z, x)
	R = np.array([x, y, z]).transpose()

	return R

def define_SE3(R, p):
    SE3 = np.identity(4)
    SE3[0:3, 0:3] = R
    SE3[0:3, 3] = p
    return SE3

