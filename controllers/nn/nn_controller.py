from controllers.base_controller import Controller


class NeuralNetworkController(Controller):
    def __init__(self, env):
        Controller.__init__(self, env)
        self.obs = self.env.reset()

    def _do_update(self, dt):
        print('computer in charge now')
        return self.predict()


    def predict(self):
        raise NotImplementedError()

