import math
import numpy as np

from .aggrevate import AggreVaTe


class UPMSLearning(AggreVaTe):

    def __init__(self, env, teacher, learner, explorer, safety_coefficient,
                 horizon, episodes):
        AggreVaTe.__init__(self, env, teacher, learner, explorer,
                                   horizon, episodes)
        self.explorer = explorer
        self._teacher_uncertainty = math.inf
        self._learner_uncertainty = math.inf
        self._safety_coefficient = safety_coefficient

    def preferential_coefficient(self, uncertainty):
        return 1. - np.tanh(self._safety_coefficient * uncertainty)

    def non_preferential_coefficient(self, uncertainty):
        return np.tanh(self._safety_coefficient * uncertainty)

    def _mix(self):

        teacher_preference = self.preferential_coefficient(self._teacher_uncertainty)
        learner_preference = self.preferential_coefficient(self._learner_uncertainty)
        normalization_factor = teacher_preference + learner_preference
        # rationality
        mixing_proportions = [teacher_preference / normalization_factor, learner_preference / normalization_factor]
        if math.isnan(mixing_proportions[0]) and math.isnan(mixing_proportions[1]): # impossibility
            self._on_emergency_action()
            return None

        selected_policy = np.random.choice(a=[self.teacher, self.learner], p=mixing_proportions)
        return selected_policy

    # \mathcal{E}^\prime
    def _mix_exploration(self):
        teacher_preference = self.preferential_coefficient(self._teacher_uncertainty)

        exploration_control = np.random.choice(a=[self.teacher, self.explorer],
                                               p=[teacher_preference, 1. - teacher_preference])

        return exploration_control

    def _act(self, observation):
        learner_preference = self.preferential_coefficient(self._learner_uncertainty)
        control_policy = np.random.choice(
            a=[self._mix(), self._mix_exploration()],
            p=[learner_preference, 1. - learner_preference]
        )
        control_action = control_policy.predict(observation)

        self._query_expert(control_policy, control_action, observation)

        return control_action

    def _on_emergency_action(self):
        print('emergency action applied')