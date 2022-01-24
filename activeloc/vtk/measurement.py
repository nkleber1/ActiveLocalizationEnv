# Imports
import numpy as np
import vtk
from scipy.ndimage.filters import gaussian_filter1d
# Relative imports
from .utils import get_cmap, sample_rotations


class LaserMeasurements:
    def __init__(self, num, renderer=None):
        # Number of lasers
        self.num = num
        self.renderer = renderer
        # Initialize at origin
        self._x = np.zeros(3)
        self.endpoints = np.zeros((3, num))

    @property
    def renderer(self):
        return self._renderer

    @renderer.setter
    def renderer(self, ren):
        self._renderer = ren
        if ren is not None:
            # Uniquely colour each laser ray based on a uniform colour map
            colors = get_cmap(self.num)
            self._lines = [vtk.vtkLineSource() for _ in range(self.num)]
            # Add to VTK scene
            for ix, line in enumerate(self._lines):
                ren.add_object(line,
                               color=colors[ix % self.num],
                               pipeline=True)

    @property
    def endpoints(self):
        return self._endpoints

    @endpoints.setter
    def endpoints(self, points):
        self._endpoints = points
        if self._renderer is not None:
            for i in range(self.num):
                self._lines[i].SetPoint1(self._x)
                self._lines[i].SetPoint2(points[:, i])

    def reset(self):
        self.endpoints = np.tile(self._x.reshape(3, 1), (1, self.num))

    def update(self, lasers, pose, mesh):
        '''
        Cast laser rays from the given sensor array pose and return points of intersection.
        Note: Returns point at max laser_range if no intersection is found.
        '''
        # Get endpoints of line segment along pointlaser direction
        epoints = pose['x'][:, None] + lasers.relative_endpoints(pose['q'])
        intersections = np.empty_like(epoints)
        # Perform intersections for each laser
        for i in range(lasers.num):
            intersections[:, i] = mesh.intersect(pose['x'], epoints[:, i])
        # Store a copy of current position
        np.copyto(self._x, pose['x'])
        self.endpoints = intersections
        return np.linalg.norm(intersections - pose['x'][:, None], axis=0)
