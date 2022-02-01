import gym, gym.spaces, gym.utils, gym.utils.seeding
from gym import spaces
from math import pi
import numpy as np
import time
from active_localization_env.robot_dynamics.resources import Robot


class RobotDynamics:
    def __init__(self, seed, render=True):
        # init render
        self.isRender = render

        # make robot
        self.robot = Robot(render)
        self.start_pose = None
        self.joint_states = None

        # same seed for robot, bullet_env and vtk_env
        self.robot.np_random = seed

    def reset(self):
        """
        Samples a random robot pose as a starting position
        :return: height and orientation of the end effector
        """
        # sample new robot pose and apply if possible.
        pose = self.robot.sample_pose()
        self.start_pose, self.joint_states = self.robot.reset(pose['x'], pose['q'])
        h_mm = self.start_pose['x'][2]
        q = self.start_pose['q']
        return h_mm, q

    def close(self):  # TODO do we need that?
        """
        Disconnect client
        """
        self.robot.close()

    def step(self, a):  # TODO self-measurement
        """
        Performs an action with the robot
        and determines the actual orientation of the end effector
        and its deviation from the original position.
        :param a: action 3*[-1, 1] (new orientation in normed euler angels)
        :return: new orientation and deviation from the start position
        """
        new_pose, self.joint_states = self.robot.apply_action(a)
        pos_noise = self.get_noise(new_pose)
        return new_pose['q'], pos_noise

    def get_noise(self, curr_pose):
        t = self.start_pose['x']
        c = curr_pose['x']
        return c - t



