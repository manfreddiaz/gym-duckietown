import tensorflow as tf
from ._baseline import Autoencoder

LATENT_SIZE = 8


class DenoisingAutoencoder(Autoencoder):

    def __init__(self, x, latent_size=LATENT_SIZE, loss_function='mse', noise=0.1):
        self.noise = noise
        Autoencoder.__init__(self, x=x, latent_size=latent_size, loss_function=loss_function)

    def _modify_input(self):
        self.x = self.x + tf.random_normal(shape=tf.shape(self.x), stddev=0.01)

    def _encoder(self, x):
        nn = tf.layers.dense(x, self.latent_size, activation=tf.nn.relu)
        return nn

    def _decoder(self, z):
        nn = tf.layers.dense(z, units=self.x.get_shape().as_list()[1])
        return nn
