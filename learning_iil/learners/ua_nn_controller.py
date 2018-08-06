from learning_iil.learners import NeuralNetworkController


class UncertaintyAwareNNController(NeuralNetworkController):

    def __init__(self, env, learner, input_shape=(None, 120, 160, 3), output_shape=(None, 2), batch_size=16,
                 storage_location='./model.ckpt', training=True):
        NeuralNetworkController.__init__(self, env, learner, input_shape, output_shape,
                                         batch_size, storage_location, training)

    def predict(self, observation):
        prediction = self.leaner.predict([observation])
        print(prediction)
        return prediction[0], prediction[1]
