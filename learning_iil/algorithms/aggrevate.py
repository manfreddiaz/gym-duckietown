import numpy as np
from learning_iil.algorithms.dagger import DAggerLearning


class AggreVaTeLearning(DAggerLearning):

    def __init__(self, env, teacher, learner, horizon, episodes, alpha=0.99):
        DAggerLearning.__init__(self, env, teacher, learner, horizon, episodes, alpha)
        self._select_breakpoint()

    def _select_breakpoint(self):
        self.break_point = np.random.randint(1, self.horizon)  # t in AggreVaTeSampling()
        print('t: {}'.format(self.break_point))

    def _do_update(self, dt):
        if self.horizon_count < self.break_point:
            control_policy = self._mix_policy()
        elif self.horizon_count >= self.break_point:  # FIXME: exploration step
            control_policy = self.primary
        # TODO: Some performance/clarity may be added below
        self._aggregate(self.env.unwrapped.render_obs(), self.primary._do_update(dt))
        return control_policy._do_update(dt)

    def _on_episode_done(self):
        DAggerLearning._on_episode_done(self)
        self._select_breakpoint()

    def _on_episode_learning_done(self):
        self.dataset.clear()
