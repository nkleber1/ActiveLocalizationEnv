'''
Common utility functions.
'''
# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.spatial.transform import Rotation


def get_cmap(num_points):
    '''
    Create a uniform colourmap of n colours.
    '''
    cm = plt.get_cmap('hsv')
    return np.array(
        [np.array(cm(i / num_points)[:-1]) for i in range(num_points)])


def sample_rotations(rot, angle, size=1, strategy='conical'):
    '''
    Sample random rotations around a given quaternion rotation.
    '''
    if strategy == 'euler':
        # Add uniform noise in each Euler angle of given rotation
        rpy = np.random.uniform(-angle, angle, (size, 3))
        noise = Rotation.from_euler('xyz', rpy)
    elif strategy == 'conical':
        # Sample rotations in a conical region around given rotation
        theta = np.random.uniform(0, angle, size=size)
        w = np.cos(theta / 2)
        x, y, z = np.sin(theta / 2) * random_unit_vectors(size)
        noise = Rotation.from_quat(np.column_stack([x, y, z, w]).squeeze(),
                                   normalized=True)
    else:
        raise ValueError("Supported strategies are ['euler', 'conical']")
    # Add noise
    return rot * noise


def random_unit_vectors(size=1):
    '''
    Generate a randomly oriented unit vector.
    ref: http://mathworld.wolfram.com/SpherePointPicking.html
    '''
    z = np.random.uniform(-1, 1, size=size)
    theta = np.random.uniform(0, 2 * np.pi, size=size)
    d = np.sqrt(1 - z * z)
    return np.array([d * np.cos(theta), d * np.sin(theta), z])


def normalize(v):
    return v / np.linalg.norm(v)

def rewards2return(rew, gamma):
    ret = 0
    for r in reversed(rew):
        ret = r + gamma*ret
    return ret

def vectors_to_rot(a, b):
    '''
    Get a right handed rotation matrix from two vectors.
    ref: On the Continuity of Rotation Representations in Neural Networks (2019)
    '''
    v1 = normalize(a)
    v2 = normalize(b - np.dot(b, v1) * v1)
    v3 = np.cross(v1, v2)
    return np.column_stack((v1, v2, v3))


def histogram(Z, lasers, bin_size):
    '''
    Accumulate a set of laser measurements into a histogram.
    Arguments:
        bin_size: Histogram bin size in mm/bin
    '''
    # Histogram
    nsamples, nrays, nlasers = Z.shape
    nbins = int(lasers.range // bin_size)
    # Accumulate values for several rays
    Z_ix = (Z // bin_size).astype(int)
    in_range = (Z_ix < nbins)
    # Discretize measurement space
    hist = np.zeros((nsamples, nlasers, nbins))
    # Uniform probability in all bins for out of range measurements
    num_out_range = nrays - in_range.sum(axis=1)
    hist += num_out_range[:, :, None] / (nbins * nrays)
    # Fill corresponding index in histogram
    indices = (np.where(in_range)[0], np.where(in_range)[2], Z_ix[in_range])
    np.add.at(hist, indices, 1 / nrays)
    # Normalized sigma in units of bin size
    sigma_norm = lasers.sigma / bin_size
    # Add gaussian measurement noise - to have a standard deviation as in real point-laser
    pz = gaussian_filter1d(hist, axis=-1, sigma=sigma_norm)
    return pz


def cov2corr(S):
    '''
    Convert covariance matrix to correlation matrix.
    Also returns the standard deviations of each dimension.
    '''
    D = np.sqrt(np.diag(S))
    P = (S / D).T / D
    return D, P


def corr2cov(D, P):
    '''
    Convert correlation matrix and standard deviation values to covcariance
    matrix.
    '''
    S = (P * D).T * D
    return S
