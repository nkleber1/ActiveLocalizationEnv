'''
Laser array configuration i.e. directions in which lasers are pointing.

Configuration: columns represent pointlaser directions in sensor frame

Single pointlaser along X axis
self._directions = np.array([[1, 0, 0]]).T

Single pointlaser along Y axis
self._directions = np.array([[0, 1, 0]]).T
Two lasers along X, Y axes
self._directions = np.array([[1, 0, 0], [0, 1, 0]]).T

Three lasers along X, Y, Z axes
self._directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T

Spherical lasers
theta = np.arange(-180, 180, 30) * np.pi / 180
phi = np.arange(-90, 90, 30) * np.pi / 180
self._directions = np.array(
     [[np.cos(p) * np.cos(t),
       np.cos(p) * np.sin(t),
       np.sin(p)] for p in phi for t in theta]).T
'''


import numpy as np


class Lasers:
    def __init__(self, num_lasers):
        if num_lasers == 1:
            # Single pointlaser along X axis
            self._directions = np.array([[1, 0, 0]]).T
        elif num_lasers == 2:
            # Two lasers along X, Y axes
            self._directions = np.array([[1, 0, 0], [0, 1, 0]]).T
        else:
            # Three lasers along X, Y, Z axes
            self._directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T

        # Number of lasers
        self.num = self._directions.shape[1]
        # Range
        self.range = 5000.  # (mm)
        # Measurement noise
        self.sigma = 20.  # (mm)

    def relative_endpoints(self, q):
        '''
        Get endpoints of laser rays based on given laser array orientation.
        '''
        rotation_matrix = q.as_matrix()
        endpoints = self.range * np.dot(rotation_matrix, self._directions)
        return endpoints

    def direction(self, q, step):
        '''
        If new orientation should be kept after each measurement.
        This can be called in step() function of pointlaser_env.py
        '''

        if step == 0:
            self._directions = np.array([[1, 0, 0]]).T
            return self._directions
        else:
            rotation_matrix = q.as_dcm()
            self._directions = np.dot(rotation_matrix, self._directions)
            self._directions = self._directions / np.linalg.norm(self._directions)
            return self._directions
