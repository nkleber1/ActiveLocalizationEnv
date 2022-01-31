import os
import time
import gym
import gym.spaces
import gym.utils
import numpy as np
from math import sqrt
import pybullet
import pybullet_data
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R

MAX_DISTANCE = 100
EULER_RANGE = np.array([np.pi, np.pi / 2, np.pi])


class Robot:
    """
    Base class for mujoco .xml based agents.
    """
    def __init__(self, args):
        self.args = args

        # init bullet client
        self.physicsClientId = -1
        self._p = None
        self.isRender = args.renderRobot
        self.setup_client()

        # Define action space
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]))

        # Robot set-up information
        path = os.path.dirname(__file__)
        self.model_urdf = path + '/mabi_urdf/mabi.urdf'
        self.robotID = self._p.loadURDF(self.model_urdf,
                                        basePosition=[0, 0, 0],
                                        baseOrientation=[0, 0, 0, 1],
                                        useFixedBase=1,
                                        flags=pybullet.URDF_USE_SELF_COLLISION | pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)  # TODO what is URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS?

        # Cataloguing and init all parts and joints
        self.parts = dict()
        self.jdict = dict()
        self.ordered_joints = list()
        for j in range(self._p.getNumJoints(self.robotID)):
            self._p.setJointMotorControl2(self.robotID, j, pybullet.POSITION_CONTROL, positionGain=0.1,
                                          velocityGain=0.1, force=0)
            joint_info = self._p.getJointInfo(self.robotID, j)
            joint_name = joint_info[1].decode("utf8")
            part_name = joint_info[12].decode("utf8")

            # Generate body part object
            self.parts[part_name] = BodyPart(self._p, part_name, self.robotID, j)

            if joint_name[:8] != "jointfix":
                self.jdict[joint_name] = Joint(self._p, joint_name, self.robotID, j)
                self.ordered_joints.append(self.jdict[joint_name])

        # Parts
        self.shoulder = self.parts["SHOULDER"]
        self.arm = self.parts["ARM"]
        self.elbow = self.parts["ELBOW"]
        self.forearm = self.parts["FOREARM"]
        self.wrist_1 = self.parts["WRIST_1"]
        self.wrist_2 = self.parts["WRIST_2"]
        self.end = self.parts["EE_FORCETORQUESENSOR"]
        self.ray_x = self.parts["ray_x"]
        self.ray_y = self.parts["ray_y"]
        self.ray_z = self.parts["ray_z"]
        self.transmitter = self.parts["transmitter"]
        self.receiver_x = self.parts["receiver_x"]
        self.receiver_y = self.parts["receiver_y"]
        self.receiver_z = self.parts["receiver_z"]

        # Joints
        self.shoulder_rot_joint = self.jdict["SH_ROT"]
        self.shoulder_fle_joint = self.jdict["SH_FLE"]
        self.elbow_fle_joint = self.jdict["EL_FLE"]
        self.elbow_rot_joint = self.jdict["EL_ROT"]
        self.wrist_fle_joint = self.jdict["WR_FLE"]
        self.wrist_rot_joint = self.jdict["WR_ROT"]
        self.laser_x = self.jdict["laser_x"]
        self.laser_y = self.jdict["laser_y"]
        self.laser_z = self.jdict["laser_z"]
        self.flex_joints = [self.shoulder_rot_joint.ID, self.shoulder_fle_joint.ID, self.elbow_fle_joint.ID,
                            self.elbow_rot_joint.ID, self.wrist_fle_joint.ID, self.wrist_rot_joint.ID]

        # Laser-Senor
        self.sensor = Sensor(self._p, self.robotID, self.transmitter.ID, self.receiver_x.ID, self.receiver_y.ID,
                             self.receiver_z.ID)
        # self.lasers = Lasers(self._p, self.args)
        # Placeholders
        self.endEffectorPosition = None
        self.endEffectorOrientation = None
        self.joint_states = None
        self.target_pos = None
        self.target_q = None

    def reset(self, pos=None, q=None):
        '''
        Bring the robot end_effector in target pose
        :param pos: target end-effector starting position
        :param q: target end-effector starting orientation
        :return: starting state (correct, actual_q, joint_states)
        '''
        # Connect to Bullet-Client (if no client is connected)
        if self.physicsClientId < 0:
            self.setup_client()

        # move end_effector to target
        self.target_pos = pos if pos is not None else [0.7, 0, 1.3]
        q = q if q is not None else [0, 0, 0]
        self.eef_to_target(q)
        return self.get_state()

    def apply_action(self, a):
        '''
        Simulate action on robot. (1) Check if action is possible. (2) Get actually reached orientation. (3) Get joint
        states of the robot arm.
        :param a: orientation in normalise euler angels 3*[-1, 1]
        :return: current state (correct, actual_q, joint_states)
        '''
        self.eef_to_target(a)
        return self.get_state()

    def eef_to_target(self, q):
        '''
        Move robot arme to target pose. Try to keep target_position and reach target_orientation.
        :param q: orientation in normalise euler angels 3*[-1, 1]
        '''
        euler = q * EULER_RANGE
        self.target_q = self._p.getQuaternionFromEuler([euler[0], euler[1], euler[2]])
        self.joint_states = self._p.calculateInverseKinematics(self.robotID, self.transmitter.ID,
                                                               targetOrientation=self.target_q,
                                                               targetPosition=self.target_pos,
                                                               residualThreshold=0.001,
                                                               maxNumIterations=100000)
        pybullet.setJointMotorControlArray(self.robotID, self.flex_joints, pybullet.POSITION_CONTROL,
                                           targetPositions=self.joint_states)
        self.step_simulation()

    def get_state(self):
        '''
        Get state of the robotic arm. (1) Check if action is possible (deviate nor more than threshold from the target
        position and do not take measurement against owen arm.). (2) Get actually reached orientation. (3) Get joint
        states of the robot arm.
        :return: current state (correct, actual_q, joint_states)
        '''
        # self-measurement
        measure_successful = self.sensor.measure_successful
        curr_q = self.wrist_2.get_orientation()

        # kept position
        t = self.target_pos
        c = self.transmitter.get_position()
        pos_noise = c - t

        # correct (no self-measurement)
        correct = all(measure_successful)

        # actual end-effector orientation
        curr_q = self._p.getEulerFromQuaternion(curr_q)

        return correct, pos_noise, curr_q, self.joint_states

    def step_simulation(self):
        for _ in range(self.args.n_step):
            self._p.stepSimulation()
            time.sleep(self.args.sleep)

    def sample_pose(self):
        # TODO actually sample
        pos = [0.6, 0, 1.2]
        q = np.random.uniform(-1, 1, 3)
        return {'x': pos, 'q': q}

    def setup_client(self):
        '''
        Set up BulletClient, Gravity, Plane  and setts PhysicsEngineParameter and debugging features.
        '''
        if self.isRender:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()
        self.physicsClientId = self._p._client
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.81)
        self._p.loadURDF("plane.urdf")
        self._p.setDefaultContactERP(0.9)
        self._p.setPhysicsEngineParameter(fixedTimeStep=self.args.time_step * self.args.frame_skip,
                                          numSolverIterations=self.args.num_solver_iterations,
                                          numSubSteps=self.args.frame_skip)

    def close(self):
        '''
        Disconnect client
        '''
        if self.physicsClientId >= 0:
            if self.physicsClientId >= 0:
                self._p.disconnect()
        self.physicsClientId = -1

    def add_debug_parameter(self):
        self.x = self._p.addUserDebugParameter("x", -1, 1, 0)
        self.y = self._p.addUserDebugParameter("y", -1, 1, 0)
        self.z = self._p.addUserDebugParameter("z", -1, 1, 0)

    def get_debug_parameter(self):
        x = self._p.readUserDebugParameter(self.x)
        y = self._p.readUserDebugParameter(self.y)
        z = self._p.readUserDebugParameter(self.z)
        return x, y, z


class BodyPart:
    def __init__(self, bullet_client, body_name, robot_id, body_part_id):  # TODO why is body_name needed?
        self.robotID = robot_id
        self.body_name = body_name
        self._p = bullet_client
        self.ID = body_part_id

    def get_pose(self):
        pos, q, _, _, _, _ = self._p.getLinkState(self.robotID, self.ID)
        return [pos, q]

    def get_position(self):
        return np.array(self.get_pose()[0])

    def get_orientation(self):
        return np.array(self.get_pose()[1])


class Joint:
    def __init__(self, bullet_client, joint_name, robot_id, joint_id):
        self.robotID = robot_id
        self._p = bullet_client
        self.ID = joint_id
        self.joint_name = joint_name

        joint_info = self._p.getJointInfo(self.robotID, self.ID)
        self.jointType = joint_info[2]
        self.lowerLimit = joint_info[8]
        self.upperLimit = joint_info[9]
        self.jointHasLimits = self.lowerLimit < self.upperLimit
        self.jointMaxVelocity = joint_info[11]

    def current_relative_position(self):
        pos, vel = self.get_state()
        if self.jointHasLimits:
            pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
            pos = 2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit)

        if self.jointMaxVelocity > 0:
            vel /= self.jointMaxVelocity
        elif self.jointType == 0:  # JOINT_REVOLUTE_TYPE
            vel *= 0.1
        else:
            vel *= 0.5
        return pos, vel

    def get_state(self):
        x, vx, _, _ = self._p.getJointState(self.robotID, self.ID)
        return x, vx

    def get_position(self):
        x, _ = self.get_state()
        return x

    def get_orientation(self):
        _, r = self.get_state()
        return r


class Sensor:
    def __init__(self, bullet_client, robot_id, transmitter_id, receiver_x_id, receiver_y_id, receiver_z_id):
        self._p = bullet_client
        self.robotID = robot_id
        self.transmitterID = transmitter_id
        self.receiver_x_id = receiver_x_id
        self.receiver_y_id = receiver_y_id
        self.receiver_z_id = receiver_z_id
        self.laser_visual = False

    def _measure(self, receiver):
        pos_transmitter = self._p.getLinkState(self.robotID, self.transmitterID)[0]
        pos_receiver = self._p.getLinkState(self.robotID, receiver)[0]
        return self._p.rayTest(pos_transmitter, pos_receiver)[0][2] * 10

    def measure(self):
        x_distance = self._measure(self.receiver_x_id)
        y_distance = self._measure(self.receiver_y_id)
        z_distance = self._measure(self.receiver_z_id)
        return [x_distance, y_distance, z_distance]

    def _get_relative_endpoint(self, receiver):
        pos_transmitter = np.array(self._p.getLinkState(self.robotID, self.transmitterID)[0])
        pos_receiver = np.array(self._p.getLinkState(self.robotID, receiver)[0])
        return pos_receiver - pos_transmitter

    def get_relative_endpoints(self):
        relative_endpoint_x = self._get_relative_endpoint(self.receiver_x_id)
        relative_endpoint_y = self._get_relative_endpoint(self.receiver_y_id)
        relative_endpoint_z = self._get_relative_endpoint(self.receiver_z_id)
        return relative_endpoint_x, relative_endpoint_y, relative_endpoint_z

    def _measure_successful(self, receiver):
        pos_transmitter = self._p.getLinkState(self.robotID, self.transmitterID)[0]
        pos_receiver = self._p.getLinkState(self.robotID, receiver)[0]
        return self._p.rayTest(pos_transmitter, pos_receiver)[0][0] != self.robotID

    @property
    def measure_successful(self):
        x_collision = self._measure_successful(self.receiver_x_id)
        y_collision = self._measure_successful(self.receiver_y_id)
        z_collision = self._measure_successful(self.receiver_z_id)
        return [x_collision, y_collision, z_collision]


# class Lasers:
#     def __init__(self, bullet_client, args):
#         self._p = bullet_client
#         if args.num_lasers == 1:
#             # Single pointlaser along X axis
#             self._directions = np.array([[1, 0, 0]]).T
#         elif args.num_lasers == 2:
#             # Two lasers along X, Y axes
#             self._directions = np.array([[1, 0, 0], [0, 1, 0]]).T
#         else:
#             # Three lasers along X, Y, Z axes
#             self._directions = np.array([[0, 0, -1], [0, -1, 0], [1, 0, 0]])
#         # Number of lasers
#         self.num = self._directions.shape[1]
#         # Range
#         self.range = 5  # (m)
#         # Initialize at origin
#         self._x = np.zeros(3)
#
#     def relative_endpoints(self, q):
#         '''
#         Get endpoints of laser rays based on given laser array orientation.
#         '''
#         q = R.from_quat(q)
#         rotation_matrix = q.as_matrix()
#         endpoints = self.range * np.dot(rotation_matrix, self._directions)
#         return endpoints.T
#
#     def update(self, pose, mesh):
#         '''
#         Cast laser rays from the given sensor array pose and return points of intersection.
#         Note: Returns point at max laser_range if no intersection is found.
#         '''
#         # Get endpoints of line segment along pointlaser direction
#         epoints = pose['x'][:, None] + self.relative_endpoints(pose['q'])
#         distance = np.zeros(self.num)
#         # Perform intersections for each laser
#         for i in range(self.num):
#             # intersections[:, i] = mesh.intersect(pose['x'], epoints[:, i])
#             distance[i] = self._p.rayTest(pose['x'], epoints[:, i])[0][2] * self.range
#         # Store a copy of current position
#         np.copyto(self._x, pose['x'])
#         return distance  # np.linalg.norm(intersections - pose['x'][:, None], axis=0)
