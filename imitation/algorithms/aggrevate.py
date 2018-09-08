import math
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
        if self._episode == 0:
            control_policy = self.teacher
        else:
            if self._current_horizon < self.t:
                control_policy = self._mix()
            elif self._current_horizon == self.t:
                control_policy = self.explorer
            else:
                control_policy = self.teacher

        control_action = control_policy.predict(observation, [self._episode, None])

        if isinstance(control_action, tuple):
            control_action, uncertainty = control_action # if we have uncertainty as input, we do not record it

        if control_policy == self.learner:
            self.learner_action = control_action
            self.learner_uncertainty = uncertainty # it might but it wont
        else:
            self.learner_action = None
            self.learner_uncertainty = math.inf

        self._query_expert(control_policy, control_action, observation)

        self._active_policy = control_policy == self.teacher

        return control_action

    def _query_expert(self, control_policy, control_action, observation):
        if control_policy == self.teacher:
            if isinstance(self.teacher_action, tuple):
                self.teacher_action, self.teacher_uncertainty = self.teacher_action # if we have uncertainty as input, we do not record it

            if self.teacher_action is not None:
                self._aggregate(observation, self.teacher_action)
                self.teacher_queried = True
            else:
                self.teacher_queried = False

    def _on_sampling_done(self):
        self.t = np.random.randint(0, self._horizon)
