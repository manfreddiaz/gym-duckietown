import numpy as np
import tensorflow as tf
from .._layers.autoencoders.denoising import DenoisingAutoencoder
from learning_iil.learners.models.tf.tf_online_learner import TensorflowOnlineLearner

from .._layers import resnet_1, resnet_1_dropout, resnet_0, resnet_2

tf.set_random_seed(1234)

lamb = 1


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
        model = tf.map_fn(lambda frame: tf.image.resize_images(frame, (60, 80)), self.state_tensor)
        model = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), model)
        model = resnet_1(model, keep_prob=1.0)

        model = tf.layers.dense(model, units=64, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

        denoising_autoencoder = DenoisingAutoencoder(model, latent_size=24, noise=0.1)
        #
        if not self.training:
            self.vector_field = tf.reduce_mean(denoising_autoencoder.decoder - model)
            #     self.log_density = tf.reduce_mean((denoising_autoencoder.decoder - model) / 0.01)
            #     self.hessian = tf.reduce_mean(tf.subtract(tf.gradients(denoising_autoencoder.decoder, model), 1.))
            self.fortified_loss = denoising_autoencoder.loss
        else:
            self.fortified_loss = denoising_autoencoder.loss

        model = tf.layers.dense(denoising_autoencoder.decoder, units=32, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

        model = tf.layers.dense(model, self.action_tensor.shape[1])

        with tf.name_scope('losses'):
            loss = tf.reduce_mean(tf.square(model - self.action_tensor), axis=1)
            tf.summary.scalar('regression', tf.reduce_mean(loss))
            loss = tf.reduce_mean(loss + self.fortified_loss)
            tf.summary.scalar('total_loss', loss)

        return [model], loss

    def get_optimizer(self, loss):
        return tf.train.AdagradOptimizer(1e-3).minimize(loss, global_step=self.global_step)

