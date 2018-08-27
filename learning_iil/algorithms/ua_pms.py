import math
import numpy as np

from learning_iil.algorithms import AggreVaTeLearning


class UPMSLearning(AggreVaTeLearning):

    def __init__(self, env, teacher, learner, explorer, safety_coefficient,
                 horizon, episodes, starting_position, starting_angle):
        AggreVaTeLearning.__init__(self, env, teacher, learner, explorer,
                                   horizon, episodes, starting_position, starting_angle)
        self.explorer = explorer
        self._teacher_uncertainty = math.inf
        self._learner_uncertainty = math.inf
        self._safety_coefficient = safety_coefficient

    def control_coefficient(self, uncertainty):
        return 1. - np.tanh(self._safety_coefficient * uncertainty)

    def exploration_coefficient(self, uncertainty):
        return np.tanh(self._safety_coefficient * uncertainty)

    def _active_policy(self):
        if self._current_episode == 0:
            return self.primary

        teacher_preference = self.control_coefficient(self._teacher_uncertainty)
        learner_preference = self.control_coefficient(self._learner_uncertainty)
        normalization_factor = teacher_preference + learner_preference

        # rationality
        mixing_proportions = [teacher_preference / normalization_factor, learner_preference / normalization_factor]
        if math.isnan(mixing_proportions[0]) and math.isnan(mixing_proportions[1]): # impossibility
            self._emergency_action()
            return None
        else:
            selected_policy = np.random.choice(a=[self.primary, self.secondary], p=mixing_proportions)
            return selected_policy

    def _active_exploration(self):
        teacher_preference = self.control_coefficient(self._teacher_uncertainty)

        exploration_control = np.random.choice(a=[self.primary, self.explorer],
                                               p=[teacher_preference, 1. - teacher_preference])

        return exploration_control

    def _select_breakpoint(self):
        self.break_point = math.floor(self._horizon * self.exploration_coefficient(self._learner_uncertainty))

    def _exploit(self, observation):
        control_action = None

        teacher_action, self._teacher_uncertainty = self.primary._do_update(self._current_episode)
        learner_action, self._learner_uncertainty = self.secondary._do_update(observation)
        control_policy = self._active_policy()

        if control_policy == self.primary:
            control_action = teacher_action

        elif control_policy == self.secondary:
            control_action = learner_action

        return control_policy, control_action

    def _explore(self, observation):
        control_action = None

        teacher_action, self._teacher_uncertainty = self.primary._do_update(self._current_episode)
        explorer_action, self._learner_uncertainty = self.explorer._do_update(observation)
        control_policy = self._active_exploration()

        if control_policy == self.primary:
            control_action = teacher_action
        elif control_policy == self.explorer:
            control_action = explorer_action

        return control_policy, control_action

    def _do_update(self, dt):
        observation = self.env.unwrapped.render_obs()

        control_action = None
        control_policy = None

        # t = f(U_learner)
        self._select_breakpoint()  # FIXME: working with old uncertainty that does not gets updated when exploring

        if self._current_horizon <= self.break_point:
            control_policy, control_action = self._exploit(observation)
        elif self._current_horizon > self.break_point:
            control_policy, control_action = self._explore(observation)

        if control_policy is not None:
            self._on_expert_input(control_policy, control_action, observation)

        return control_action

    def _emergency_action(self):
        # self.secondary.enabled = False  # disable the learner, allows the teacher to control back
        print('emergency action applied')

    def reset(self):
        self.primary.reset()
        self.secondary.reset()

        self.primary.enabled = True
        self.secondary.enabled = True

        AggreVaTeLearning.reset(self)

    def _on_episode_done(self):
        self._learn()
        self.reset()

    def _on_learning_done(self):
        pass