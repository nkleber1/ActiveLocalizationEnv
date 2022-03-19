import numpy as np
import gym
import math
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from typing import Union
import os

class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env: Union[gym.Env, VecEnv], **kwargs):
        super().__init__(eval_env, **kwargs)

    def _on_rollout_end(self) -> bool:
        """
        This event is triggered before updating the policy.
        """
        #print('eval', self.eval_env.envs[0])
        eval_env = self.eval_env.envs[0]
        vol_mean = np.mean(np.asarray(eval_env.volume_per_eps))
        max_axis_mean = np.mean(np.asarray(eval_env.maxaxis_per_eps))
        gt_dist_mean = np.mean(np.asarray(eval_env.gt_dist))
        if not math.isnan(gt_dist_mean):
            print(gt_dist_mean)
            self.model.logger.record('eval/Average Volume', vol_mean)
            self.model.logger.record('eval/Average Max Axis', max_axis_mean)
            self.model.logger.record('eval/Dist to Ground Truth', gt_dist_mean)
        eval_env.clear_loggig_lists()
        return True


class LoggingCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, rollouts_per_summary=1, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.rollouts_per_summary = rollouts_per_summary
        self.rollout_cnt = 0
        self.vol_mean = 0
        self.gt_dist_mean = 0
        self.max_axis_mean = 0
        self.env = None

    def _on_rollout_end(self) -> bool:
        """
        This event is triggered before updating the policy.
        """
        #print('train', self.model.get_env().envs[0])
        if not self.env:
            self.env = self.model.get_env().envs[0]
        self.rollout_cnt += 1
        self.vol_mean += np.mean(np.asarray(self.env.volume_per_eps))
        self.max_axis_mean += np.mean(np.asarray(self.env.maxaxis_per_eps))
        self.gt_dist_mean += np.mean(np.asarray(self.env.gt_dist))
        if self.rollout_cnt == self.rollouts_per_summary:
            self.model.logger.record('rollout/Average Volume', self.vol_mean)
            self.model.logger.record('rollout/Average Max Axis', self.max_axis_mean)
            self.model.logger.record('rollout/Dist to Ground Truth', self.gt_dist_mean)
            self.rollout_cnt = 0
            self.vol_mean = 0
            self.gt_dist_mean = 0
            self.max_axis_mean = 0
        self.env.clear_loggig_lists()
        return True

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
