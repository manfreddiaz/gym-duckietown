from learning_iil.learners import NeuralNetworkController


class UncertaintyAwareNNController(NeuralNetworkController):

    def predict(self, observation):
        action, uncertainty = self.leaner.predict(observation)
        return action, uncertainty
