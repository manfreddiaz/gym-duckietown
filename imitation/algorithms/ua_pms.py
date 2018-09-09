import math
import numpy as np

from imitation.algorithms import DAgger
from .aggrevate import AggreVaTe


class UPMS(DAgger):

    def __init__(self, env, teacher, learner, explorer, safety_coefficient, horizon, episodes):
        DAgger.__init__(self, env, teacher, learner, horizon, episodes)
        self.explorer = explorer

        self._safety_coefficient = safety_coefficient

    def _normalize_uncertainty(self, uncertainty):
        return np.sum(self.learner_uncertainty) / uncertainty.shape[0]

    def _preferential_coefficient(self, uncertainty):
        return 1. - np.tanh(self._safety_coefficient * uncertainty)

    def _non_preferential_coefficient(self, uncertainty):
        return np.tanh(self._safety_coefficient * uncertainty)

    def _mix(self):
        alpha_p = self._preferential_coefficient(self.teacher_uncertainty)
        alpha_q = self._preferential_coefficient(self._normalize_uncertainty(self.learner_uncertainty))
        normalization = alpha_p + alpha_q
        # rationality
        mixing_proportions = [alpha_p / normalization, alpha_q / normalization]
        # impossibility
        if math.isnan(mixing_proportions[0]) and math.isnan(mixing_proportions[1]):
            self._on_impossible_selection()
            return None

        selected_policy = np.random.choice(a=[self.teacher, self.learner], p=mixing_proportions)
        return selected_policy

    # \mathcal{E}^\prime
    def _mix_exploration(self):
        teacher_preference = self._preferential_coefficient(self.teacher_uncertainty)

        exploration_control = np.random.choice(a=[self.teacher, self.explorer],
                                               p=[teacher_preference, 1. - teacher_preference])

        return exploration_control

    def _act(self, observation):
        if self._episode == 0:
            control_policy = self.teacher
        else:
            learner_preference = self._preferential_coefficient(self._normalize_uncertainty(self.learner_uncertainty))
            control_policy = np.random.choice(
                a=[self._mix(), self._mix_exploration()],
                p=[learner_preference, 1. - learner_preference]
            )
        control_action, uncertainty = control_policy.predict(observation, [self._episode, None])

        self._query_expert(control_policy, control_action, uncertainty, observation)

        self._active_policy = control_policy == self.teacher

        return control_action

    def _on_impossible_selection(self):
        print('emergency action applied')