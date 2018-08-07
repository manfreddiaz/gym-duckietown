from .dagger import DAggerLearning


class SafeDAggerLearning(DAggerLearning):

    def __init__(self, env, teacher, learner, threshold, horizon, episodes, starting_position, starting_angle, alpha=0.99):
        DAggerLearning.__init__(self, env, teacher,
                                learner, horizon, episodes, starting_position, starting_angle, alpha)
        self.threshold = threshold

    def _active_policy(self):
        pass