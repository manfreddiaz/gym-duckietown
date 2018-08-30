import numpy as np
import tensorflow as tf
from learning_iil.learners.parametrizations.tf.tf_online_learner import TensorflowOnlineLearner

from .._layers import resnet_1, resnet_1_dropout


class MonteCarloDropoutResnetOneRegression(TensorflowOnlineLearner):

    def explore(self, state, horizon=1):
        pass

    def __init__(self, optimizer, name=None, samples=25, dropout_prob=0.9):
        TensorflowOnlineLearner.__init__(self)
        self.name = name
        self.samples = samples
        self.dropout_prob = dropout_prob
        self.optimizer = optimizer

    def predict(self, state, horizon=1):
        regression = TensorflowOnlineLearner.predict(self, np.repeat(state, self.samples, axis=0))
        regression = regression[0]
        return np.squeeze(np.mean(regression, axis=1)), np.sum(np.var(regression, axis=1))

    def architecture(self):
        model = tf.map_fn(lambda frame: tf.image.resize_images(frame, (60, 80)), self.state_tensor)
        model = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), model)

        model = resnet_1(model, keep_prob=1.0)
        model = tf.layers.dense(model, units=64, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        # model = tf.nn.dropout(model, keep_prob=0.5)
        model = tf.layers.dense(model, units=32, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        model = tf.nn.dropout(model, keep_prob=self.dropout_prob)

        model = tf.layers.dense(model, self.action_tensor.shape[1])
        with tf.name_scope('losses'):
            loss = tf.losses.mean_squared_error(model, self.action_tensor)
            tf.summary.scalar('mse', loss)

        return [model], loss

    def get_optimizer(self, loss):
        return self.optimizer.minimize(loss, global_step=self.global_step)

