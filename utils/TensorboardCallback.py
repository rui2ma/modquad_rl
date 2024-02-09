from stable_baselines3.common.callbacks import BaseCallback
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.reward = 0

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        self.logger.record("x", self.training_env.get_attr('x'))
        self.logger.record("y", self.training_env.get_attr('y'))
        self.logger.record("z", self.training_env.get_attr('z'))
        return True

#
# class TensorboardCallback(BaseCallback):
#     def __init__(self, verbose=1):
#         super(TensorboardCallback, self).__init__(verbose)
#         self.cum_rew_1 = 0
#         self.cum_rew_2 = 0
#
#     def _on_rollout_end(self) -> None:
#         self.logger.record("rollout/cum_rew_1", self.cum_rew_1)
#         self.logger.record("rollout/cum_rew_2", self.cum_rew_2)
#
#         # reset vars once recorded
#         self.cum_rew_1 = 0
#         self.cum_rew_2 = 0
#
#     def _on_step(self) -> bool:
#         self.cum_rew_1 += self.training_env.get_attr("reward_1")[0]
#         self.cum_rew_2 += self.training_env.get_attr("reward_2")[0]
#         return True