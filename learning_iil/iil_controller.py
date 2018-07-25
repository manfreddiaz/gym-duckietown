from controllers import SharedController


class IILController(SharedController):

    def __init__(self, env, teacher, learner, horizon):
        SharedController.__init__(self, env, teacher, learner)

    def _do_update(self, dt):
        pass

    def step(self, action):
        pass
