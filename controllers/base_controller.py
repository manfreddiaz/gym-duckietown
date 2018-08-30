import types

import pyglet
import sys
import yaml


class Controller:

    def __init__(self, env, refresh_rate=1/30):
        self.enabled = True
        self.env = env
        self.mapping = None
        self.refresh_rate = refresh_rate

    # dynamically extends a controller capabilities by redirecting calls to the extender.
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

    def configure(self):
        pass

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

    # basic capability: reset the environment
    def reset(self):
        self.env.reset()
        self.env.render()

    # basic capability: exit the environment
    def exit(self):
        self.close()
        self.env.close()
        sys.exit(0)

    def open(self):
        pyglet.clock.schedule_interval(self.update, self.refresh_rate)
        self.env.unwrapped.window.push_handlers(self)

    def close(self):
        pyglet.clock.unschedule(self.update)
        self.env.unwrapped.window.remove_handlers(self)
