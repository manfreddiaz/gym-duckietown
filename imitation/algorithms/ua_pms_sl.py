from .ua_pms import UPMS


class UPMSSelfLearning(UPMS):

    def _self_learning(self, observation, control_action):
        self._aggregate(observation, control_action)

