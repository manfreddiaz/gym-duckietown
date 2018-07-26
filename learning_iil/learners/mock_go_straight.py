from controllers import TensorflowNNController


class MockStraightController(TensorflowNNController):

    def __init__(self, env):
        TensorflowNNController.__init__(self, env)

    def predict(self):
        return [0.2, 0.0]

    def learn(self, dataset):
        pass
