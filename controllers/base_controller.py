import types

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

    # dynamically extends a controller capabilities by redirecting the call to the extender.
    @staticmethod
    def extend_capabilities(extender, controller, actions):
        # expects a dict with name:action
        for action in actions:
            setattr(controller, action, types.MethodType(actions[action], extender))

    @staticmethod
    def has_capability(controller, capability):
        return getattr(controller, capability, None) is not None

    @staticmethod
    def invoke_capability(controller, capability, arguments):
        capability_method = getattr(controller, capability, None)
        if capability is function:
            return capability_method(**arguments)

    def _initialize(self):
        pyglet.clock.schedule_interval(self.update, self.refresh_rate)
        self.env.unwrapped.window.push_handlers(self)

    def load_mapping(self, path):
        with open(path) as mf:
            self.mapping = yaml.load(mf)

    def update(self, dt):
        if self.enabled:
            action = self._do_update(dt=dt)
            if action is not None:
                self.step(action=action)

    def _do_update(self, dt):
        raise NotImplementedError

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
