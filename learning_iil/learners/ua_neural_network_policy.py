from learning_iil.learners import NeuralNetworkPolicy


class UANeuralNetworkPolicy(NeuralNetworkPolicy):

    def __init__(self, env, parametrization, input_shape=(None, 120, 160, 3), output_shape=(None, 2), batch_size=16,
                 storage_location='./model.ckpt', training=True):
        NeuralNetworkPolicy.__init__(self, env, parametrization, input_shape, output_shape,
                                     batch_size, storage_location, training)

    def predict(self, observation):
        prediction = self.parametrization.predict([observation])
        print(prediction)
        return prediction[0], prediction[1]

    def reset(self):
        pass
