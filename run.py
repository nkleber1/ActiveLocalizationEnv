import os
from stable_baselines3 import PPO
import argparse
from datetime import datetime
from SB3_extensions import CombinedExtractor
from active_localization_env import make_envs, LoggingCallback
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

''' Training configuration parameters '''
TOTAL_TIMESTEPS = 5000000


class Config(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Reinforcement Learning Active Localization')
        # Log Directories Setup
        parser.add_argument('--load', type=str, default=None, help='load a saved model')
        parser.add_argument('--log_name', type=str, default=datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'),
                            help='Log Folder Name')
        # Learning
        parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate of Adam optimizer')
        parser.add_argument('--discount_factor', type=float, default=0.99)
        # Reward
        parser.add_argument('--reward', type=str, default='original', metavar='N',
                            choices=['original', 'simple'], help='...')
        # Observation space
        parser.add_argument('--map_obs', type=str, default='lidar', metavar='N',
                            choices=['grid_encodings', 'point_encodings', '3d_encodings', 'lidar_encodings',
                                     'point_cloud', 'point_cloud_3d', 'lidar', 'depth'],
                            help='...')  # TODO description
        parser.add_argument('--map_size', type=int, default=None, help='...')
        parser.add_argument('--num_lasers', type=int, default=1, help='...')
        parser.add_argument('--use_mean_pos', type=bool, default=True, help='Use Mean Position estimate in the state space')
        parser.add_argument('--use_measurements', type=bool, default=True, help='Use Laser Measurements in the state space')
        parser.add_argument('--use_map', type=bool, default=True, help='Use map info in the state space')
        parser.add_argument('--use_map_height', type=bool, default=True, help='Use map height in the state space')
        parser.add_argument('--use_joint_states', type=bool, default=True, help='Use use_joint_states in the state space')
        # pyBullet
        parser.add_argument('--robot_dynamics', type=bool, default=False, help='...')
        parser.add_argument('--render_robot', type=bool, default=False, help='...')
        self.args = parser.parse_args()

    def get_arguments(self):
        return self.args


def main():
    args = Config().get_arguments()

    # make envs
    env, eval_env = make_envs(args)

    # Initialize Callback List
    log_callback = LoggingCallback()
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model', log_path='./logs/results',
                                 eval_freq=5120)
    callback = CallbackList([log_callback, eval_callback])

    # use the CombinedExtractor
    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
    )

    model = PPO('MultiInputPolicy', env, verbose=1,  policy_kwargs=policy_kwargs,
                gamma=args.discount_factor, ent_coef=0.01,
                tensorboard_log='logs/tensorboard', n_steps=1024,
                learning_rate=args.lr, vf_coef=0.5, max_grad_norm=0.5,
                _init_setup_model=True, seed=None)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, tb_log_name=args.log_name)

    model.save(os.path.join('logs/weights', args.log_name))
    env.save(os.path.join('logs/envs', args.log_name, "env.pkl"))
    print('\nTraining Finished Successfully !')

    # print('----- EVAL -----')
    # # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=20.)
    # obs = eval_env.reset()
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = eval_env.step(action)
    #     eval_env.render()
    #     if done:
    #         obs = eval_env.reset()


if __name__ == '__main__':
    main()
