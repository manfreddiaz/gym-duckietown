import pyglet
import sys
import yaml


class Controller:

    def __init__(self, env, refresh_rate=0.1):
        self.enabled = True
        self.env = env
        self.mapping = None
        self.refresh_rate = refresh_rate
        self._initialize()

    def _initialize(self):
        pyglet.clock.schedule_interval(self.update, self.refresh_rate)
        self.env.unwrapped.window.push_handlers(self)

    def load_mapping(self, path):
        with open(path) as mf:
            self.mapping = yaml.load(mf)

    def update(self, dt):
        if self.enabled:
            self._do_update(dt=dt)

    def _do_update(self, dt):
        raise NotImplementedError

    def _has_capability(self, capability):
        return getattr(self, capability, None) is not None

    # action
    def step(self, action):
        response = self.env.step(action)
        self.env.render()
        return response

    # action
    def reset(self):
        self.env.reset()
        self.env.render()

    def exit(self):
        self.close()
        self.env.close()
        sys.exit(0)

    def close(self):
        pyglet.clock.unschedule(self.update)
        self.env.unwrapped.window.remove_handlers(self)
