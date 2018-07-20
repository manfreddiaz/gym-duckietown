import pyglet
import yaml


class Controller:

    def __init__(self, env):
        self.enabled = True
        self.env = env
        self.mapping = None
        self.initialize()

    def initialize(self):
        pyglet.clock.schedule_interval(self.update, 0.1)
        self.env.unwrapped.window.push_handlers(self)

    def load_mapping(self, path):
        with open(path) as mf:
            self.mapping = yaml.load(mf)

    def update(self, dt):
        if self.enabled:
            self._do_update(dt=dt)

    def _do_update(self, dt):
        raise NotImplementedError

    # action
    def record(self):
        pass

    # action
    def step(self, action):
        response = self.env.step(action)
        self.env.render()
        return response

    # action
    def reset(self):
        self.env.reset()
        self.env.render()


