from controllers.base_controller import Controller


class NeuralNetworkController(Controller):
    def __init__(self, env):
        Controller.__init__(self, env)
        self.obs = self.env.reset()

    def _do_update(self, dt):
        return self.predict()
        # self.observation, _, _, _ = self.step(action=action)
        print('computer in charge now')

    def predict(self):
        raise NotImplementedError()

