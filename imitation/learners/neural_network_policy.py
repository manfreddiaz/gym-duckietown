import numpy as np
from tqdm import tqdm


class NeuralNetworkPolicy():

    def __init__(self, parametrization, optimizer=None, input_shape=(None, 120, 160, 3), output_shape=(None, 2),
                 batch_size=16, epochs=10, storage_location=None, training=True):

        self.parametrization = parametrization
        self.batch_size = batch_size
        self.epochs = epochs

        if training:
            self.parametrization.prepare_for_train(input_shape, output_shape, optimizer, storage_location)
        else:
            self.parametrization.prepare_for_test(input_shape, output_shape, storage_location)

    def optimize(self, observations, expert_actions):
        observations = np.array(observations)
        actions = np.array(expert_actions)
        data_size = observations.shape[0]

        for _ in tqdm(range(self.epochs)):
            for iteration in range(0, data_size, self.batch_size):
                batch_observations = observations[iteration:iteration + self.batch_size]
                batch_actions = actions[iteration:iteration + self.batch_size]
                self.parametrization.train(batch_observations, batch_actions)
            if iteration + self.batch_size < data_size:
                remainder = data_size - iteration
                batch_observations = observations[iteration:remainder]
                batch_actions = actions[iteration:remainder]
                self.parametrization.train(batch_observations, batch_actions)


    def predict(self, observation, metadata):
        return self.parametrization.test([observation])

    def save(self):
        self.parametrization.commit()

    def close(self):
        self.parametrization.close()
