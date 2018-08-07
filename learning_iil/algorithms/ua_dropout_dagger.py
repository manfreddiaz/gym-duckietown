import math

from .dagger import DAggerLearning


class DropoutDAggerLearning(DAggerLearning):
    def __init__(self, env, teacher, learner,
                 threshold, horizon, episodes, starting_position, starting_angle, alpha=0.99):
        DAggerLearning.__init__(self, env, teacher, learner,
                                horizon, episodes, starting_position, starting_angle, alpha)
        self.threshold = threshold
        self.learner_uncertainty = math.inf

    def _active_policy(self):
        if self._current_episode == 0:  # check DAgger definition (initial policy equals expert)
            return self.primary

        if self.learner_uncertainty > self.threshold:
            return self.primary
        else:
            return self.secondary

    def _do_update(self, dt):
        observation = self.env.unwrapped.render_obs()
        learner_action, self.learner_uncertainty = self.secondary._do_update(observation)
        # print(learner_action, self.learner_uncertainty)
        control_policy = self._active_policy()

        if control_policy == self.primary:
            control_action, _ = self.primary._do_update(observation)
        else:
            control_action = learner_action

        self._on_expert_input(control_policy, control_action, observation)

        return control_action
