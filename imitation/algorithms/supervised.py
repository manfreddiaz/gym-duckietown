from .iil_learning import InteractiveImitationLearning


class SupervisedLearning(InteractiveImitationLearning):
    def __init__(self, env, teacher, learner, horizon, episodes, starting_position, starting_angle):
        InteractiveImitationLearning.__init__(self, env, teacher, learner, horizon, episodes)

    def _mix(self):
        return self.teacher
