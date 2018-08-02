import numpy as np
import tensorflow as tf
from .._layers.autoencoders.denoising import DenoisingAutoencoder
from learning_iil.learners.models.tf.tf_online_learner import TensorflowOnlineLearner

from .._layers import resnet_1, resnet_1_dropout, resnet_0, resnet_2

tf.set_random_seed(1234)

lamb = 0.05


class FortifiedResnetOneRegression(TensorflowOnlineLearner):
    def explore(self, state, horizon=1):
        pass

    def __init__(self, name=None):
        self.name = name
        self.fortified_loss = None
        TensorflowOnlineLearner.__init__(self)

    def predict(self, state, horizon=1):
        regression = TensorflowOnlineLearner.predict(self, state)
        return np.squeeze(regression), self.fortified_loss

    def architecture(self):
        model = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.state_tensor)
        model = resnet_1(model)
        denoising_autoencoder = DenoisingAutoencoder(model, latent_size=10)
        self.fortified_loss = denoising_autoencoder.loss

        model = tf.layers.dense(denoising_autoencoder.decoder, units=64, activation=tf.nn.relu)

        model = tf.layers.dense(model, self.action_tensor.shape[1])
        with tf.name_scope('losses'):
            loss = tf.reduce_mean(tf.square(model - self.action_tensor), axis=1)
            loss = tf.reduce_mean(loss + lamb * self.fortified_loss)
            tf.summary.scalar('mse', loss)

        return [model], loss

    def get_optimizer(self, loss):
        return tf.train.AdagradOptimizer(1e-3).minimize(loss, global_step=self.global_step)

