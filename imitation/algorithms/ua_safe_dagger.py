from .dagger import DAgger


class SafeDAggerLearning(DAgger):

    def __init__(self, env, teacher, learner, threshold, horizon, episodes, alpha=0.5):
        DAgger.__init__(self, env, teacher,
                        learner, horizon, episodes, alpha)
        self.threshold = threshold

    def _mix(self):
        pass