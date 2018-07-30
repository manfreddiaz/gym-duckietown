import math
import numpy as np

from learning_iil.algorithms import AggreVaTeLearning


class UPMSLearning(AggreVaTeLearning):

    def __init__(self, env, teacher, learner, explorer, horizon, episodes, starting_position, starting_angle):
        AggreVaTeLearning.__init__(self, env, teacher, learner, horizon, episodes, starting_position, starting_angle)
        self.explorer = explorer
        self.teacher_uncertainty = math.inf
        self.learner_uncertainty = math.inf

    @staticmethod
    def control_coefficient(uncertainty):
        return 1. - np.tanh(uncertainty)

    @staticmethod
    def exploration_coefficient(uncertainty):
        return np.tanh(uncertainty)

    def _select_policy(self):
        teacher_preference = UPMSLearning.control_coefficient(self.teacher_uncertainty)
        learner_preference = UPMSLearning.control_coefficient(self.learner_uncertainty)
        normalization_factor = teacher_preference + learner_preference

        # rationality
        mixing_proportions = [teacher_preference / normalization_factor, learner_preference / normalization_factor]
        if math.isnan(mixing_proportions[0]) and math.isnan(mixing_proportions[1]): # impossibility
            self._emergency_action()
        else:
            selected_policy = np.random.choice(a=[self.primary, self.secondary], p=mixing_proportions)
            return selected_policy

        return None

    def _select_exploration(self):
        teacher_preference = UPMSLearning.control_coefficient(self.teacher_uncertainty)

        exploration_control = np.random.choice(a=[self.primary, self.explorer],
                                               p=[teacher_preference, 1. - teacher_preference])

        return exploration_control

    def _select_breakpoint(self):
        self.break_point = math.floor(self.horizon * UPMSLearning.exploration_coefficient(self.learner_uncertainty))

    def _exploit(self, observation):
        control_action = None

        teacher_action, self.teacher_uncertainty = self.primary._do_update(observation)
        learner_action, self.learner_uncertainty = self.secondary._do_update(observation)
        control_policy = self._select_policy()

        if control_policy == self.primary:
            control_action = teacher_action
        elif control_policy == self.secondary:
            control_action = learner_action

        return control_action, teacher_action

    def _explore(self, observation):
        control_action = None

        teacher_action, self.teacher_uncertainty = self.primary._do_update(observation)
        learner_action, self.learner_uncertainty = self.explorer._do_update(observation)
        control_policy = self._select_exploration()
        if control_policy is not None:
            control_action, _ = control_policy._do_update(observation)

        return control_action, teacher_action

    def _do_update(self, dt):
        observation = self.env.unwrapped.render_obs()

        control_action = None
        teacher_action = None

        self._select_breakpoint()  # t = f(U_learner)
        if self.horizon_count <= self.break_point:
            control_action, teacher_action = self._exploit(observation)
        elif self.horizon_count > self.break_point:
            control_action, teacher_action = self._explore(observation)

        if teacher_action is not None:
            self._aggregate(observation, teacher_action)

        return control_action

    def _emergency_action(self):
        self.secondary.enabled = False  # disable the learner, allows the teacher to control back
        print('emergency action applied')

    def reset(self):
        self.primary.enabled = True
        self.secondary.enabled = True
        AggreVaTeLearning.reset(self)

    def _on_episode_done(self):
        self._learn()
