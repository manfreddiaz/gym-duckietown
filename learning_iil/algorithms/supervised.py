from learning_iil.iil_learning import InteractiveImitationLearning


class SupervisedLearning(InteractiveImitationLearning):
    def __init__(self, env, teacher, learner, horizon, episodes, starting_position, starting_angle):
        InteractiveImitationLearning.__init__(self, env, teacher, learner, horizon, episodes, starting_position, starting_angle)

    def _active_policy(self):
        return self.primary

    def _on_episode_done(self):
        pass

    def _on_process_done(self):
        self._learn()
        InteractiveImitationLearning._on_process_done(self)

