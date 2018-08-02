import numpy as np
from ..iil_learning import InteractiveImitationLearning

np.random.seed(1234)


class DAggerLearning(InteractiveImitationLearning):
    def __init__(self, env, teacher, learner, horizon, episodes, starting_position, starting_angle, alpha=0.99):
        InteractiveImitationLearning.__init__(self, env, teacher, learner,
                                              horizon, episodes, starting_position, starting_angle)

        # from DAgger
        self.alpha = alpha
        # internal count
        self.alpha_episode = self.alpha

    def _select_policy(self):
        if self.episodes_count == 0:  # check DAgger definition (initial policy equals expert)
            return self.primary
        expert_control_decay = np.random.choice(
            a=[self.primary, self.secondary],
            p=[self.alpha_episode, 1. - self.alpha_episode]
        )
        return expert_control_decay

    def _on_episode_done(self):
        # decay expert probability of control after each episode
        self.alpha_episode = self.alpha_episode ** self.episodes_count
        InteractiveImitationLearning._on_episode_done(self)
