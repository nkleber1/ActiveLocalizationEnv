'''
Filter describing how belief is updated based on new measurements.
'''
# Imports
import numpy as np
# Relative imports
from .utils import sample_rotations, histogram


class CustomBayesFilter:
    def __init__(self,
                 lasers,
                 mesh=None,
                 num_samples=30,
                 num_casts=20,
                 angle_noise=5 * np.pi / 180,
                 sample_strategy='conical'):
        # Initialization
        self._mesh = mesh
        self._lasers = lasers
        # Number of Monte-Carlo samples
        self._num_samples = num_samples
        # Minimum number of non-zero probability samples required for update
        self._min_samples = 10
        # Number of laser rays to cast
        self._num_casts = num_casts
        # Half-angle of cone within which rotation noise is sampled
        self._angle_noise = angle_noise
        self.sample_strategy = sample_strategy

    def measurement_update(self, xbel, q, z, hist_bin_size=1):
        '''
        Update belief based on a measurment 'z' from ground truth position.
        '''
        # Position samples
        samples = xbel.sample(self._num_samples)
        # Histogram bin containing true measurement
        zint = (z // hist_bin_size).astype(int)
        num_bins = int(self._lasers.range // hist_bin_size)
        in_range = (zint < num_bins)
        # Rotation samples
        noisy_rotations = sample_rotations(q,
                                           self._angle_noise,
                                           size=self._num_samples *
                                           self._num_casts,
                                           strategy=self.sample_strategy, z=z, laser_range=self._lasers.range)
        # Laser rays from sampled positions and rotations
        endpoints = samples[:, None, :, None] + self._lasers.relative_endpoints(
            noisy_rotations).reshape(
                (self._num_samples, self._num_casts, 3, self._lasers.num))
        # Intersections of laser rays with mesh
        intersections = np.empty_like(endpoints)
        for i in range(self._num_samples):
            xi = samples[i]
            for j in range(self._lasers.num):
                endpoints_ij = endpoints[i, :, :, j]
                intersections[i, :, :, j] = np.apply_along_axis(
                    lambda y: self._mesh.intersect(xi, y),
                    axis=-1,
                    arr=endpoints_ij)
        # Distance measurements
        Z = np.linalg.norm(intersections - samples[:, None, :, None], axis=2)
        # Histogram of measurement samples
        hist = histogram(Z, self._lasers, hist_bin_size)
        # Likelihood of measurement
        pxz = np.prod(hist[:, in_range, zint[in_range]], axis=1)
        # Only update belief if there are enough samples with non-zero probability
        if np.count_nonzero(pxz) > self._min_samples:
            # Update belief parameters with sample mean/covariance
            xbel.mean = np.average(samples.T, axis=1, weights=pxz)
            xbel.cov = np.cov(samples.T, aweights=pxz)
