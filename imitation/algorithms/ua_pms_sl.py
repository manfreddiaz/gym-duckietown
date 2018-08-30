from .ua_pms import UPMSLearning


class UPMSSelfLearning(UPMSLearning):

    def _on_expert_input(self, control_policy, control_action, observation):
        if control_policy == self.primary:
            expert_action = control_action
        else:
            expert_action = self.primary._do_update(observation)  # if we have uncertainty as input, we do not record it
            if isinstance(expert_action, tuple):
                expert_action, _ = expert_action

        if expert_action is not None:
            self._aggregate(observation, expert_action)
            self._expert_interventions += 1
        elif control_policy is self.secondary:
            if isinstance(control_action, tuple):
                control_action, _ = control_action
            self._aggregate(observation, control_action)  # if the expert is not in it, self-feed it
            print('self-learning')

