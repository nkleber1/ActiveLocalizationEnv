import tensorflow as tf
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, steps_per_summary, steps_per_update, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.rollouts_per_summary = steps_per_summary/steps_per_update
        self.rollout_cnt = 0
        self.vol_mean = 0
        self.gt_dist_mean = 0
        self.max_axis_mean = 0

    def _on_rollout_end(self) -> bool:
        """
        This event is triggered before updating the policy.
        """
        env = self.model.get_env().envs[0]
        self.rollout_cnt += 1
        self.vol_mean += np.mean(np.asarray(env.volume_per_eps))
        self.max_axis_mean += np.mean(np.asarray(env.maxaxis_per_eps))
        self.gt_dist_mean += np.mean(np.asarray(env.gt_dist))
        if self.rollout_cnt == self.rollouts_per_summary:
            summary = tf.Summary(value=[tf.Summary.Value(tag='Epoch/Average Volume',
                                                         simple_value=(self.vol_mean/self.rollout_cnt))])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
            summary = tf.Summary(value=[tf.Summary.Value(tag='Epoch/Average Max Axis',
                                                         simple_value=(self.max_axis_mean/self.rollout_cnt))])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
            summary = tf.Summary(value=[tf.Summary.Value(tag='Epoch/Dist to Ground Truth',
                                                         simple_value=(self.gt_dist_mean/self.rollout_cnt))])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
            self.rollout_cnt = 0
            self.vol_mean = 0
            self.gt_dist_mean = 0
            self.max_axis_mean = 0
        env.volume_per_eps = []
        env.maxaxis_per_eps = []
        env.gt_dist = []
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
