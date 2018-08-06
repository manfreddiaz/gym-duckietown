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

    def __init__(self, name=None, noise=1e-2):
        TensorflowOnlineLearner.__init__(self)
        self.name = name
        self.fortified_loss = None
        self.noise = noise
        self.vector_field = None
        self.vector_field_value = None

    def predict(self, state, horizon=1):
        regression = TensorflowOnlineLearner.predict(self, state)
        return np.squeeze(regression), np.abs(self.vector_field_value)

    def architecture(self):
        model = tf.map_fn(lambda frame: tf.image.resize_images(frame, (60, 80)), self.state_tensor)
        model = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), model)

        if self.training:
            dropout_prob = 0.5
        else:
            dropout_prob = 1.0

        model = resnet_1(model, keep_prob=dropout_prob)

        model = tf.layers.dense(model, units=64, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

        model = tf.layers.dense(model, units=32, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
        denoising_autoencoder = DenoisingAutoencoder(model, latent_size=12, noise=self.noise)
        #
        score_function = denoising_autoencoder.decoder - model
        self.vector_field = tf.reduce_mean(score_function) / self.noise
        # self.hessian = tf.reduce_mean(tf.subtract(tf.gradients(denoising_autoencoder.loss, model), 1))
        self.fortified_loss = denoising_autoencoder.loss

        model = tf.layers.dense(denoising_autoencoder.decoder, self.action_tensor.shape[1])

        with tf.name_scope('losses'):
            loss = tf.reduce_mean(tf.square(model - self.action_tensor), axis=1)
            tf.summary.scalar('regression', tf.reduce_mean(loss))
            loss = tf.reduce_mean(loss + self.fortified_loss)
            tf.summary.scalar('total_loss', loss)

        return [model], loss

    def get_optimizer(self, loss):
        return tf.train.AdagradOptimizer(1e-3).minimize(loss, global_step=self.global_step)

