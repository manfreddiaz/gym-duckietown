from learning_iil.iil_controller import InteractiveImitationLearning


class SupervisedLearning(InteractiveImitationLearning):
    def __init__(self, env, teacher, learner, horizon, episodes, starting_position, starting_angle):
        InteractiveImitationLearning.__init__(self, env, teacher, learner, horizon, episodes, starting_position, starting_angle)

    def _select_policy(self):
        return self.primary

    def _on_training_done(self):
        self._learn()
        InteractiveImitationLearning._on_training_done(self)


