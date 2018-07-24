import types
from controllers.base_controller import Controller


class SharedController(Controller):

    def __init__(self, env, human_controller, nn_controller):
        self.human_controller = human_controller
        self.nn_controller = nn_controller
        Controller.__init__(self, env=env)

    def _initialize(self):
        Controller.extend_capabilities(self, self.human_controller, {'share': self.share})
        self.nn_controller.enabled = False
        Controller._initialize(self)

    def _do_update(self, dt):
        self.human_controller.update(dt)
        self.nn_controller.update(dt)

    def share(self, caller):
        self.human_controller.enabled = not self.human_controller.enabled
        self.nn_controller.enabled = not self.nn_controller.enabled

    def close(self):
        self.human_controller.close()
        self.nn_controller.close()