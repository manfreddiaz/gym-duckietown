from controllers.nn.nn_controller import NeuralNetworkController


class TensorflowNNController(NeuralNetworkController):
    def __init__(self, env):
        NeuralNetworkController.__init__(self, env)

    def predict(self):
        print('predicting')

