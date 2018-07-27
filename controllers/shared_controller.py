import types
from controllers.base_controller import Controller


class SharedController(Controller):

    def _do_update(self, dt):
        observation = self.env.unwrapped.render_obs()
        if self.primary.enabled:
            return self.primary._do_update(observation)
        elif self.secondary.enabled:
            return self.secondary._do_update(observation)

        return None  # consistence

    def __init__(self, env, primary, secondary, shared=[True, False]):
        Controller.__init__(self, env=env)
        self.primary = primary
        self.secondary = secondary
        self.shared = shared

    def configure(self):
        Controller.extend_capabilities(self, self.primary, {'share': self.share})

        self.primary.configure()
        self.secondary.configure()

        self.primary.enabled = self.shared[0]
        self.secondary.enabled = self.shared[1]

    # extended capability
    def share(self, _):
        self.primary.enabled = not self.primary.enabled
        self.secondary.enabled = not self.secondary.enabled
        print('primary: {}, secondary: {}'.format(self.primary.enabled, self.secondary.enabled))