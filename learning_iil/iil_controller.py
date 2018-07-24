from controllers import SharedController


class InteractiveImitationLearningController(SharedController):

    def __init__(self, env, teacher, learner):
        SharedController.__init__(self, env, teacher, learner)

    def share(self, _):

        SharedController.share(self, caller=_)

    def step(self, action):
        pass
