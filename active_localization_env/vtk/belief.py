'''
Belief class for storing covariance ellipsoid parameters.
'''
# Imports
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2, multivariate_normal
# Relative Imports
from .visualization import Ellipsoid, Text


class PositionBelief:
    def __init__(self, mu=np.zeros(3), sigma=np.eye(3), renderer=None):
        # Setup rendering
        self.renderer = renderer
        # Initialize parameters
        self.mean = mu
        self.cov = sigma

    @property
    def renderer(self):
        return self._renderer

    @renderer.setter
    def renderer(self, ren):
        self._renderer = ren
        if ren is not None:
            # Scale ellipsoid to contain 95% probability volume
            confidence_interval = 0.95
            scale = np.sqrt(chi2.ppf(confidence_interval, df=3))
            self._ellipsoid = Ellipsoid(ren, scale)
            self._uncertainty_str = Text(ren)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mu):
        self._mean = mu
        if self._renderer is not None:
            self._ellipsoid.position = mu

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, sigma):
        self._cov = sigma
        if self._renderer is not None:
            # Eigenvalues, Eigenvectors
            l, v = np.linalg.eigh(sigma)
            v[:, 2] = np.cross(v[:, 0], v[:, 1])
            rot = Rotation.from_dcm(v)
            # Ellipsoid axes lengths
            self._ellipsoid.axes = np.sqrt(l)
            # Ellipsoid axes directions
            self._ellipsoid.orientation = rot
            # Update text
            self._uncertainty_str.text = 'Uncertainty: {}'.format(
                self.uncertainty())

    def sample(self, size=1):
        return multivariate_normal.rvs(self._mean, self._cov, size).squeeze()

    def uncertainty(self, metric='max_eigval'):
        '''
        Uncertainty of belief (inverse of information content)
        '''
        if metric == 'entropy':
            return multivariate_normal(self._mean, self._cov).entropy()
        elif metric == 'det':
            return np.sqrt(np.cbrt(np.linalg.det(self._cov)))
        elif metric == 'trace':
            return np.sqrt(np.trace(self._cov))
        elif metric == 'max_eigval':
            return np.sqrt(np.linalg.eigvalsh(self._cov)[-1])
        else:
            raise ValueError(
                "Supported metrics are ['entropy', 'det', 'trace', 'max_eigval']"
            )

    def __str__(self):
        return 'Mean: {}\nCovariance:\n'.format(self._mean) + '\t' + str(
            self._cov
        ).replace(
            '\n', '\n\t'
        ) + '\nUncertainty:\n\tEntropy: {:.2f}\n\tDet    : {:.2f}\n\tTrace  : {:.2f}\n\tMax Eig: {:.2f}'.format(
            self.uncertainty('entropy'), self.uncertainty('det'),
            self.uncertainty('trace'), self.uncertainty())
