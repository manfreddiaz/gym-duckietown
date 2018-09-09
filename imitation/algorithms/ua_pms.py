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

    # Rational Policy Mixing
    def _rpm(self, policy_p, policy_p_uncertainty, policy_q, policy_q_uncertainty):
        alpha_p = self._preferential_coefficient(policy_p_uncertainty)
        alpha_q = self._preferential_coefficient(policy_q_uncertainty)
        # consistency
        normalization = alpha_p + alpha_q
        p_mix, q_mix = alpha_p / normalization, alpha_q / normalization
        # impossibility
        if math.isnan(p_mix) and math.isnan(q_mix):
            self._on_impossible_selection()
            return None
        # rationality
        return np.random.choice(a=[policy_p, policy_q], p=[p_mix, q_mix])

    # Preferential Policy Mixing
    def _ppm(self, policy_p, policy_p_uncertainty, policy_s):
        alpha_p = self._preferential_coefficient(policy_p_uncertainty)

        return np.random.choice(a=[policy_p, policy_s],
                                               p=[alpha_p, 1. - alpha_p])

    def _mix(self):
        return self._rpm(self.teacher, self.teacher_uncertainty, self.learner, self._normalize_uncertainty(self.learner_uncertainty))

    # \mathcal{E}^\prime
    def _mix_exploration(self):
        return self._ppm(self.teacher, self.teacher_uncertainty, self.explorer)

    def _act(self, observation):
        if self._episode == 0:
            control_policy = self.teacher
        else:
            learner_preference = self._preferential_coefficient(self._normalize_uncertainty(self.learner_uncertainty))
            pi_i = self._mix()
            if pi_i is not None:
                control_policy = np.random.choice(
                    a=[pi_i, self._mix_exploration()],
                    p=[learner_preference, 1. - learner_preference]
                )
            else:
                control_policy = None

        if control_policy is not None:
            control_action, uncertainty = control_policy.predict(observation, [self._episode, None])
        else:
            control_action, uncertainty = None, math.inf

        self._query_expert(control_policy, control_action, uncertainty, observation)

        self._active_policy = control_policy == self.teacher

        return control_action

    def _on_impossible_selection(self):
        print('emergency action applied')