'''
Wrapper for VTK mesh.
'''
# Imports
import os
import numpy as np
import cv2 as cv
import vtk
from stl import mesh
from .local_info import get_local_info, correct_rotations_m1
#from vtk.util.numpy_support import vtk_to_numpy
from active_localization_env.env import MESH_FILE

class Mesh:
    def __init__(self, mesh_nr, mesh_file_dir, raycast_tol=1, renderer=None):
        # Load mesh
        mesh_file = os.path.join(mesh_file_dir, MESH_FILE + str(mesh_nr) + '.stl')
        #mesh_file, mesh, scale = transform(mesh_file=mesh_file)  # uncomment if rescaling of mesh needed
        self._mesh = self.load_stl(mesh_file)

        # Build BSP tree for ray-casting
        self.bsp_tree = vtk.vtkModifiedBSPTree()
        self.bsp_tree.SetDataSet(self._mesh)
        self.bsp_tree.BuildLocator()
        # Implicit function to find if point is inside/outside surface and at what distance
        # https://vtk.org/doc/nightly/html/classvtkImplicitFunction.html#details
        self._implicit_function = vtk.vtkImplicitPolyDataDistance()
        self._implicit_function.SetInput(self._mesh)
        # Ray casting tolerance
        self._tol = raycast_tol
        # Mesh boundaries
        bounds = self._mesh.GetBounds()
        self.min_bounds = np.array(bounds[::2])
        self.max_bounds = np.array(bounds[1::2])
        # Set-up rendering
        self.renderer = renderer

    @property
    def renderer(self):
        return self._renderer

    @renderer.setter
    def renderer(self, ren, color='LightGrey'):
        self._renderer = ren
        if ren is not None:
            # Add room mesh to scene
            room_actor = ren.add_object(self._mesh, color=color)
            # Modify visual properties
            room_actor.GetProperty().SetOpacity(0.6)
            ren.reset_camera()
            # Show mesh edges
            self._display_edges()

    def _display_edges(self):
        '''
        Highlight prominent mesh edges.
        '''
        feature_edges = vtk.vtkFeatureEdges()
        feature_edges.SetInputData(self._mesh)
        feature_edges.Update()
        # Visualize
        self._renderer.add_object(feature_edges, pipeline=True)

    @classmethod
    def load_stl(self, file):
        '''
        Load STL file into VTK mesh.
        '''
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file)
        # Read the STL file
        reader.Update()
        polydata = reader.GetOutput()
        # If there are no points in 'vtkPolyData' something went wrong
        if polydata.GetNumberOfPoints() == 0:
            raise ValueError('No point data could be loaded from ' + file)
        return polydata

    def intersect(self, p1, p2):
        '''
        Intersect line segment defined by end points with the mesh.
        '''
        points = vtk.vtkPoints()
        # Perform intersection
        code = self.bsp_tree.IntersectWithLine(p1, p2, self._tol, points, None)
        if code == 0:
            # Return p2 if no intersection is found
            return p2.copy()
        return np.array(points.GetData().GetTuple3(0))

    def is_inside(self, position):
        '''
        Check if point is inside mesh.
        '''
        return self._implicit_function.FunctionValue(position) <= 0

    def sample_position(self, dmin=0, dmax=np.inf):
        '''
        Sample a random position inside mesh which is between dmin and dmax
        distance from mesh boundary. dmin is offset from mesh, dmax is laser range.
        '''
        while True:
            position = np.random.uniform(self.min_bounds + dmin,
                                         self.max_bounds - dmin)
            signed_dist = self._implicit_function.FunctionValue(position)
            if (dmin < -signed_dist < dmax):
                return position

    def sample_position_verification(self, dmin=0, dmax=np.inf):
        while True:
            position = np.random.uniform(self.min_bounds + dmin,
                                         self.max_bounds - dmin)
            signed_dist = self._implicit_function.FunctionValue(position)
            region, _ = correct_rotations_m1(position)
            if (dmin < -signed_dist < dmax) and region > 0:
                return position


# TODO check if we need to transform
# def transform(mesh_file):
#     mesh_env = mesh.Mesh.from_file(mesh_file)
#     data = mesh_env.data
#     range = np.max(data['vectors'])-np.min(data['vectors'])
#
#     if range < 5000:  # user set range of dimension
#         args.rescale = 400  # rescale factor, can be changed depending on mesh dimensions
#         data['normals'] *= args.rescale
#         data['vectors'] *= args.rescale
#         data['attr'] *= args.rescale
#         transformed_mesh = mesh.Mesh(data)
#         mesh_path = os.path.join(args.mesh_dir, args.mesh_file + '_rescaled' + args.mesh_nr + '.stl')
#         transformed_mesh.save(mesh_path)
#     else:  # just in case the mesh is already large enough
#         args.rescale = 1
#         transformed_mesh = mesh.Mesh(data)
#         mesh_path = mesh_file
#
#     return transformed_mesh, mesh_path, args.rescale
