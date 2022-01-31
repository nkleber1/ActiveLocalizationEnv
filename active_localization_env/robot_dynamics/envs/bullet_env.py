import gym, gym.spaces, gym.utils, gym.utils.seeding
from gym import spaces
from math import pi
from vtk_pointlaser.config import Config
from vtk_pointlaser import RandomizedPointlaserEnv
from mabi_pointlaser.resources.robot import Robot
import numpy as np
import time


class BulletEnv(gym.Env):
    """
	Base class for Bullet physics simulation environments in a Scene.
	These environments create single-player scenes and behave like normal Gym environments, if
	you don't use multiplayer.
	"""

    def __init__(self, render=True):
        # get args
        self.args = Config().get_arguments()

        # init render
        self.isRender = render

        # make robot
        self.robot = Robot(self.args)
        self.action_space = self.robot.action_space

        # make vtk
        self.vtk_env = RandomizedPointlaserEnv(self.args)
        low = self.vtk_env.observation_space_low
        high = self.vtk_env.observation_space_high
        if self.args.use_joint_states:
            n_joints = len(self.robot.flex_joints)
            low = np.hstack((low, -2*pi*np.zeros(n_joints)))
            high = np.hstack((high, 2*pi*np.zeros(n_joints)))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float)


        # same seed for robot, bullet_env and vtk_env
        self.np_random = None
        self.seed(self.args.seed)

        # episode history
        self._current_step = 0
        self.episode_reward = 0

    def seed(self, seed=None):
        '''
        Set random seed for BulletEnv and MabiRobot
        '''
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.robot.np_random = self.np_random
        self.vtk_env.np_random = self.np_random
        return [seed]

    def reset(self):
        '''
        Starts a new episode, withe random robot pose.
        :return: first obs (by vtk env)
        '''
        # Set Episode info to zero
        self._current_step = 0
        self.episode_reward = 0

        # sample new robot pose and apply if possible.
        while True:
            pose = self.robot.sample_pose()
            correct, pos_noise, q, joint_states = self.robot.reset(pose['x'], pose['q'])

            # if correct:  # TODO check again
            h_mm = pose['x'][2]
            obs = self.vtk_env.reset(h_mm, q)
            if self.args.use_joint_states:
                obs = np.append(obs, joint_states)
            return obs

    def close(self):
        '''
        Disconnect client
        '''
        self.robot.close()

    def step(self, a):
        '''
        Perform a new step. (1) Check if action is possible under robot dynamics. (2) If, so hand action over to vtk_env
        :param a: action 3*[-1, 1] (new orientation in normed euler angels)
        :return: obs, reward, done, info by vtk_env (or punishment if action is impossible)
        '''
        # Increment count
        self._current_step += 1
        # Clip out of range actions
        a = np.clip(a, self.action_space.low, self.action_space.high)
        correct, pos_noise, q, joint_states = self.robot.apply_action(a)
        obs, reward, done, info = self.vtk_env.step(q, pos_noise)
        self.episode_reward += reward
        if self.args.use_joint_states:
            obs = np.append(obs, joint_states)
        return obs, reward, done, info

