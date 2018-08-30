import math

from .dagger import DAgger


class DropoutDAggerLearning(DAgger):
    def __init__(self, env, teacher, learner, threshold, horizon, episodes, alpha=0.5):
        DAgger.__init__(self, env, teacher, learner, horizon, episodes, alpha)
        self.threshold = threshold
        self.learner_uncertainty = math.inf

    def _mix(self):
        if self.learner_uncertainty > self.threshold:
            return self.teacher
        else:
            return self.learner
