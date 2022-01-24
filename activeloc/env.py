'''
OpenAI Gym environment for taking pointlaser measurements to reduce
uncertainty.
'''
# Params
MIN_MESH_OFFSET = 100  # Minimum offset from mesh boundary
POSITION_STDDEV = 50  # Initial position standard deviation
N_DISC_ACTIONS = None  # Discretization of action space
REWARD_ALPHA = 1  # Final reward weight for max eigen value reduction
USE_GOAL_REWARD = True  # Use extra  reward if goal is reached
MIN_UNCERTAINTY = 5  # Uncertainty goal
GOAL_REWARD = 100  # Value for goal achieved
HORIZON = 10  # Maximum number of steps (measurements) in one epoch

# Params for reward function  # TODO make file for reward function and add reward function based on result
USE_UNCERT_REWARD = True
UNCERT_REWARD = 0.1  # Weight for uncertainty reduction reward
USE_DIST_REWARD = True
DIST_REWARD = 0.2  # Weight for distance reduction reward
USE_EIGVAL_REWARD = True
EIGVAL_REWARD = 0.5  # eight for eigen value reduction reward
MEASUREMENT_COST = 0.0
VAR_EPS_LEN = False  # Variable Episode Length

# Params for dirs
DATASET_DIR = 'activeloc/data/train_data'
MESH_DIR = 'meshes'
MESH_FILE = 'r_map'


# Imports
import gym
import os
import numpy as np
import vtk
from scipy.spatial.transform import Rotation as R
from gym import spaces
from gym.utils import seeding
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KDTree
# Relative Imports
from .vtk import CustomBayesFilter, Lasers, LaserMeasurements, Mesh, PositionBelief, Renderer
from .vtk.utils import cov2corr
from .vtk.local_info import get_local_info, correct_rotations_m1, local_info_robot
from .point_clouds.utils import do_lidar_scan
from .point_clouds import Encoder

# Absolute range of XYZ euler angles
EULER_RANGE = np.array([np.pi, np.pi / 2, np.pi])

# To ignore 'Box bound precision' Warning
gym.logger.set_level(40)


class ActiveLocalizationEnv(gym.Env):
    def __init__(self, map_obs='point_encodings', map_size=16, num_lasers=1,
                 use_mean_pos=True, use_measurements=True, use_map=True, use_map_height=True):
        # Load mesh
        self._mesh_nr = 0
        # Initialize Laser cofiguration
        self.num_lasers = num_lasers
        self._lasers = Lasers(num_lasers)
        # Initialize belief class
        self._xbel = PositionBelief()
        # Initialize Ground Truth dict
        self._pose_gt = {}
        self._q = Rotation.from_euler('xyz', np.zeros(3))
        # Initialize Measurements class
        self._meas = LaserMeasurements(self._lasers.num)
        # Initialize Filter update class
        self._bayes_filter = CustomBayesFilter(self._lasers)
        # Maximum standard devitation
        self._max_stddev = POSITION_STDDEV
        # Minimum offset from mesh boundaries to sample positions
        self._mesh_offset = np.array([MIN_MESH_OFFSET, self._lasers.range])
        # Disable rendering
        self.rendering = False
        self.renderer = Renderer()

        self.use_mean_pos = use_mean_pos
        self.use_measurements = use_measurements
        self.use_map = use_map
        self.map_size = map_size
        self.map_obs = map_obs
        self.use_map_height = use_map_height
        self.observation_space = self.make_observation_space()
        # Numerical range of actions: Normalized rotation
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]),
                                       high=np.array([1, 1, 1]),
                                       dtype=np.float)

        self._n_disc_actions = N_DISC_ACTIONS
        # Weighing factor for final reward
        self._alpha = REWARD_ALPHA
        self.use_goal_rew = USE_GOAL_REWARD
        self.min_uncertainty = MIN_UNCERTAINTY
        self.goal_rew = GOAL_REWARD
        # store values for logging
        self.volume_per_eps = []
        self.maxaxis_per_eps = []
        self.gt_dist = []
        # Seed RNG
        self.seed()

        # Example kwargs # TODO add to Config
        encoder_args = {
            'pretrain': 'point_encoder/weights/lidar_16.pkl',
            'encoder_type': 'graph_s',
            'feat_dims': map_size,
            'k': 64,
            'dataset': 'original_uni',
            'no_vae': False,
        }
        if map_obs == 'lidar_encodings':
            self.encoder = Encoder(encoder_args)

    def _normalize_position(self, xyz):
        '''
        Linearly normalize coordinates inside mesh to range 0 and 1.
        '''
        max_dim = (self._mesh.max_bounds - self._mesh.min_bounds).max()
        min_dim = self._mesh.min_bounds.min()
        return (xyz - min_dim) / max_dim

    def _normalize_measurement(self, z):
        '''
        Linearly normalize measurements inside mesh to range 0 and 1.
        '''
        return z/self._lasers.range

    def _normalize_cov(self, cov):
        '''
        Return flat vector containing normalized standard deviations and
        correlation coefficients calculated from the covariance matrix.
        '''
        sigma, corr = cov2corr(self._xbel.cov)
        sigma_norm = sigma / self._max_stddev
        # sigma_norm = np.clip(sigma_norm, 0, 1)
        flat_corr = corr[np.triu_indices(3, k=1)]
        return np.hstack((sigma_norm, flat_corr))

    def _get_observation(self, z_norm=None):
        '''
        Normalize belief parameters to get observation.
        '''
        obs = self._normalize_cov(self._xbel.cov)
        mean_norm = self._normalize_position(self._xbel.mean) # return mean_norm
        laser_dir = np.dot(self._q.as_matrix(), self._lasers._directions).T.flatten()
        if self.use_mean_pos: obs = np.hstack((obs, mean_norm))
        if self.use_measurements: obs = np.hstack((obs, laser_dir))
        if self.use_map: obs = np.hstack((obs, self._curr_map))
        if self.use_map_height: obs = np.hstack((obs, self._curr_mesh_h))

        if z_norm:
            if self.use_measurements: obs = np.hstack((obs, z_norm.squeeze()))
        else:
            if self.use_measurements: obs = np.hstack((obs, np.zeros(self.num_lasers)))
        return obs

    def make_observation_space(self):
        # Numerical range of observations:
        # [Normalized XYZ postion, Standard Deviations, Correlations, Observations, Map Encoding, map_height]
        low = np.array([0, 0, 0, -1, -1, -1])
        high = np.array([+1, +1, +1, +1, +1, +1])
        if self.use_mean_pos:
            low = np.hstack((low, np.zeros(3)))
            high = np.hstack((high, np.ones(3)))
        if self.use_measurements:
            low = np.hstack((low, -np.ones(self.num_lasers * 4)))
            high = np.hstack((high, np.ones(self.num_lasers * 4)))
        if self.use_map:
            low = np.hstack((low, -np.inf * np.ones(self.map_size)))
            high = np.hstack((high, np.inf * np.ones(self.map_size)))
        if self.use_map_height:
            low = np.hstack((low, np.zeros(1)))
            high = np.hstack((high, np.ones(1)))
        return spaces.Box(low=low, high=high, dtype=np.float)

    def _get_reward(self):
        '''
        Return immediate reward for the step.
        '''
        # Reward is the information gain (reduction in uncertainty) where
        # uncertainty is defined as the volume of covariance ellipsoid
        uncertainty = self._xbel.uncertainty('det')
        max_eigval = self._xbel.uncertainty('max_eigval')
        dist = np.linalg.norm(self._xbel.mean-self._pose_gt['x'])
        reward = USE_UNCERT_REWARD*UNCERT_REWARD*(self._prev_uncertainty - uncertainty) + USE_DIST_REWARD*DIST_REWARD*(self._prev_dist - dist) + USE_EIGVAL_REWARD*EIGVAL_REWARD*(self._prev_max_eigval - max_eigval) - MEASUREMENT_COST
        self._prev_max_eigval = max_eigval
        self._prev_uncertainty = uncertainty
        self._prev_dist = dist
        if self._is_done():
            # Final reward is the reduction in the major axis length of the
            # covariance ellipsoid from the start of the episode
            final_reward = self._initial_maxeigval - self._xbel.uncertainty(
                'max_eigval')
            reward += self._alpha * final_reward
        if self.use_goal_rew and bool((uncertainty < self.min_uncertainty)):
            reward += self.goal_rew
        return reward

    def _is_done(self):
        '''
        Termination condition of episode.
        '''
        uncertainty = self._xbel.uncertainty('max_eigval')
        done = (self._current_step >= HORIZON)
        if VAR_EPS_LEN:
            done = bool(done or (uncertainty < MIN_UNCERTAINTY))
        if done:
            self.store_values()
        return done

    def store_values(self):
        self.volume_per_eps.append(self._xbel.uncertainty('det'))
        self.maxaxis_per_eps.append(self._xbel.uncertainty('max_eigval'))
        self.gt_dist.append(np.linalg.norm(self._xbel.mean-self._pose_gt['x']))

    @staticmethod
    def _action2rot(action):
        '''
        Get rotation from a normalized action.
        '''
        euler = action * EULER_RANGE
        return Rotation.from_euler('xyz', euler)

    def _transform_action(self, action):
        '''
        Transform to discrete actions and to euler rotations
        '''
        if self._n_disc_actions:
            n = self._n_disc_actions
            action_list = np.zeros([np.power(n+1, 3), 3])
            index = 0
            for i in range(n+1):
                ii = -1 + (2 / n) * i
                for j in range(n+1):
                    jj = -1 + (2 / n) * j
                    for k in range(n+1):
                        kk = -1 + (2 / n) * k
                        action_list[index, :] = (ii, jj, kk)
                        index += 1
            tree = KDTree(action_list, leaf_size=2)
            _, ind = tree.query(np.expand_dims(action, axis=0), k=1)
            action_disc = (action_list[ind]).squeeze()  # discrete action
            euler = action_disc * EULER_RANGE
            rot = Rotation.from_euler('xyz', euler)
            return rot, action_disc
        euler = action * EULER_RANGE
        rot = Rotation.from_euler('xyz', euler)
        return rot, action

    def step(self, action):
        # Increment count
        self._current_step += 1
        # Clip out of range actions
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Transform to discrete action and rotation
        rot, action = self._transform_action(action)
        # Convert to scipy Rotation object
        #rot = self._action2rot(action)
        self._q = rot
        self._pose_gt['q'] = rot
        # Take measurement from new orientation
        z = self._meas.update(self._lasers, self._pose_gt, self._mesh)
        z_norm = self._normalize_measurement(z)
        # Update belief
        self._bayes_filter.measurement_update(self._xbel, self._q, z)
        # Calculate reward
        reward = self._get_reward()
        obs = self._get_observation(z_norm)
        done = self._is_done()
        info = {"action": action}
        return obs, reward, done, info

    def reset(self):
        '''
        Reset environment to start a new episode.
        '''
        self._current_step = 0
        # Reset VTK measurement visualization
        self._meas.reset()

        # Import new mesh
        mesh_file_dir = os.path.join(DATASET_DIR, MESH_DIR)
        num_meshes = len(os.listdir(mesh_file_dir))
        self._mesh_nr = 1 + (self._mesh_nr%num_meshes)
        self._mesh = Mesh(self._mesh_nr, mesh_file_dir)
        self._curr_mesh_h = np.asarray([(self._mesh.max_bounds-self._mesh.min_bounds)[2]]).reshape(1)
        self._bayes_filter._mesh = self._mesh

        # Sample position
        pos = self._mesh.sample_position(*self._mesh_offset)

        # get map_encoding
        self._curr_map = self._get_map(pos)

        # Reset belief
        self._xbel.mean = pos
        # Initial diagonal covariance matrix
        self._xbel.cov = np.eye(3) * self._max_stddev**2
        # Store uncertainty for calculating reward
        self._initial_maxeigval = self._xbel.uncertainty('max_eigval')
        self._prev_uncertainty = self._xbel.uncertainty('det')
        self._prev_max_eigval = self._initial_maxeigval
        # Sample a fixed ground truth position
        self._pose_gt['x'] = self._xbel.sample()
        self._prev_dist = np.linalg.norm(self._xbel.mean-self._pose_gt['x'])
        # Reset orientation of lasers
        self._pose_gt['q'] = Rotation.from_euler('xyz', np.zeros(3))
        self._q = Rotation.from_euler('xyz', np.zeros(3))
        # Get observation from belief
        obs = self._get_observation()
        return obs

    def render(self, renderer=None, reset=False):
        if renderer == None: renderer = self.renderer
        if self.rendering == False:
            self.rendering = True
            reset = True
        if reset == True:
            self._mesh.renderer = renderer
            self._xbel.renderer = renderer
            self._meas.renderer = renderer
        renderer.render()

    def get_map(self):
        return self._mesh._map

    def _get_map(self, position):  # TODO make position self.position
        if self.map_obs in ['grid_encodings', 'point_encodings', '3d_encodings', 'point_cloud']:
            map_file = self._get_map_file()
            return np.load(map_file)
        elif self.map_obs == 'lidar_encodings':
            bsp_tree = self._mesh._bsp_tree
            point_cloud = do_lidar_scan(position, bsp_tree)
            encoding = self.encoder.encode_np(point_cloud)
            return encoding
        elif self.map_obs == 'lidar':
            bsp_tree = self._mesh._bsp_tree
            point_cloud = do_lidar_scan(position, bsp_tree, num_points=1024)
            return point_cloud

    def _get_map_file(self,):
        return os.path.join(DATASET_DIR, self.map_obs, self.map_obs + "_" +
                            str(self.map_size), MESH_FILE + str(self._mesh_nr) + '.npy')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
