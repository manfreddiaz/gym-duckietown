from learning_iil.learners import NeuralNetworkController


class UncertaintyAwareNNController(NeuralNetworkController):

    def predict(self, observation):
        prediction = self.leaner.predict([observation])
        return prediction[0], prediction[1]
