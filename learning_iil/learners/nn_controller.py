import numpy as np
import tensorflow as tf
from controllers import Controller

tf.set_random_seed(1234)


class NeuralNetworkController(Controller):

    def _do_update(self, dt):
        return self.predict(dt)

    def __init__(self, env, learner, input_shape=(None, 120, 160, 3), output_shape=(None, 2), batch_size=16,
                 storage_location='./model.ckpt', training=True):
        Controller.__init__(self, env)
        self.seen_samples = 0
        self.leaner = learner
        self.batch_size = batch_size
        if training:
            self.leaner.init_train(input_shape, output_shape, storage_location)
        else:
            self.leaner.init_test(input_shape, output_shape, storage_location)

    def learn(self, observations, expert_actions):
        data_size = len(observations)
        observations = np.array(observations)
        actions = np.array(expert_actions)
        iteration = 0
        for epoch in range(20):
            for iteration in range(0, data_size, self.batch_size):
                batch_observations = observations[iteration:iteration + self.batch_size]
                batch_actions = actions[iteration:iteration + self.batch_size]
                self.leaner.learn(batch_observations, batch_actions)
            if iteration + self.batch_size < data_size:
                remainder = data_size - iteration
                batch_observations = observations[iteration:remainder]
                batch_actions = actions[iteration:remainder]
                self.leaner.learn(batch_observations, batch_actions)

        self.seen_samples += len(observations)

    def predict(self, observation):
        action = self.leaner.predict([observation])
        return action[0]

    def save(self):
        self.leaner.commit()

    def close(self):
        Controller.close(self)
        self.leaner.close()
