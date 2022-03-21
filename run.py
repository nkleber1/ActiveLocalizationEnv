import os
from stable_baselines3 import PPO, TD3
import argparse
from datetime import datetime
from SB3_extensions import CombinedExtractor
from active_localization_env import make_envs, LoggingCallback, CustomEvalCallback
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

''' Training configuration parameters '''
TOTAL_TIMESTEPS = 5000000


class Config(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Reinforcement Learning Active Localization')
        parser.add_argument('--num_cpu', type=int, default=None, help='...')
        # Log Directories Setup
        parser.add_argument('--load', type=str, default=None, help='load a saved model')
        parser.add_argument('--log_name', type=str, default=datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M'),
                            help='Log Folder Name')

        # Learning
        parser.add_argument('--algo', type=str, default='PPO',
                            help='chose algo')
        parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate of Adam optimizer')
        parser.add_argument('--discount_factor', type=float, default=0.99)
        # Reward
        parser.add_argument('--reward', type=str, default='original', metavar='N',
                            choices=['original', 'simple'], help='...')
        # Observation space
        parser.add_argument('--map_obs', type=str, default='point_encodings', metavar='N',
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
        # Env
        parser.add_argument('--n_disc_actions', type=str, default=None,
                            help='Discretization of action space')
        parser.add_argument('--horizon', type=str, default=None,
                            help='Maximum number of steps (measurements) in one epoch')
        parser.add_argument('--min_uncertainty', type=str, default=None,
                            help='Uncertainty goal')
        parser.add_argument('--n_disc_actions', type=str, default=None,
                            help='Discretization of action space')
        # Test
        parser.add_argument('--noise_sample_strategy', type=str, default='conical', help='...')
        parser.add_argument('--batch_size', type=int, default=64, help='...')
        self.args = parser.parse_args()

    def get_arguments(self):
        return self.args


def main():
    args = Config().get_arguments()

    # make envs
    env, eval_env = make_envs(args)

    # Initialize Callback List
    log_callback = LoggingCallback()
    eval_callback = CustomEvalCallback(eval_env, best_model_save_path='./logs/best_model', log_path='./logs/results',
                                 eval_freq= 5120)
    callback = CallbackList([log_callback, eval_callback])

    # use the CombinedExtractor
    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
    )

    if args.algo == 'PPO':
        model = PPO('MultiInputPolicy', env, verbose=1,  policy_kwargs=policy_kwargs,
                    gamma=args.discount_factor, ent_coef=0.01,
                    tensorboard_log='logs/tensorboard', n_steps=1024,
                    learning_rate=args.lr, vf_coef=0.5, max_grad_norm=0.5,
                    batch_size=args.batch_size, clip_range=0.2, clip_range_vf=None, gae_lambda=0.95,
                    _init_setup_model=True, seed=None)
    else:
        model = TD3('MultiInputPolicy', env, gamma=args.discount_factor, learning_rate=args.lr, buffer_size=50000,
                    learning_starts=100, train_freq=100, gradient_steps=100, batch_size=128, tau=0.005, policy_delay=2,
                    action_noise=None, target_policy_noise=0.2, target_noise_clip=0.5, verbose=1,
                    tensorboard_log='logs/tensorboard', _init_setup_model=True, policy_kwargs=None, seed=None)
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
