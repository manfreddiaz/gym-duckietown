import numpy as np
from .dagger import DAgger


class AggreVaTe(DAgger):

    def __init__(self, env, teacher, learner, explorer, horizon, episodes, alpha=0.99):
        DAgger.__init__(self, env, teacher, learner, horizon, episodes, alpha)
        # self._select_breakpoint()
        self.break_point = None
        self.explorer = explorer
        self.t = horizon


    def _act(self, observation):
        if self._current_episode == 0:
            control_policy = self.teacher
        else:
            if self._current_horizon < self.t:
                control_policy = self._mix()
            elif self._current_horizon == self.t:
                control_policy = self.explorer
            else:
                control_policy = self.teacher

        control_action = control_policy.predict(observation)

        self._query_expert(control_policy, control_action, observation)

        self._active_policy = control_policy == self.teacher

        return control_action


    def _on_sampling_done(self):
        self.t = np.random.uniform(0, self._horizon)
