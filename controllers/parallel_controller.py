import numpy as np
from controllers import SharedController


class ParallelController(SharedController):

    def __init__(self, env, primary, secondary, mixture=np.array([0.5, 0.5])):
        SharedController.__init__(self, env, primary, secondary, shared=[True, True])
        self.mixture = mixture

    def _do_update(self, dt):
        observation = self.env.unwrapped.render_obs()

        if self.primary.enabled and self.secondary.enabled:
            primary_action = self.primary._do_update(observation)
            secondary_action =  self.secondary._do_update(observation)
            if primary_action is not None and secondary_action is not None:
                control_action = self.mixture[0] * primary_action + self.mixture[1] * secondary_action
            elif primary_action is not None:
                control_action = primary_action
            elif secondary_action is not None:
                control_action = secondary_action
            else:
                control_action = None
            return control_action
        elif self.primary.enabled:
            return  self.primary._do_update(observation)
        elif self.secondary.enabled:
            return  self.secondary._do_update(observation)

        return None  # no controller enabled.
