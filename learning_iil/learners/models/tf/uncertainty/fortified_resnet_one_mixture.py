import numpy as np
import tensorflow as tf
from learning_iil.learners.models.tf._layers.autoencoders.denoising import DenoisingAutoencoder
from learning_iil.learners.models.tf.tf_online_learner import TensorflowOnlineLearner

from .._layers import resnet_1, MixtureDensityNetwork


class FortifiedResnetOneMixture(TensorflowOnlineLearner):
    def explore(self, state, horizon=1):
        pass

    def __init__(self):
        TensorflowOnlineLearner.__init__(self)
        self.fortified_loss = None
        self.manifold_loss = None

    def predict(self, state, horizon=1):
        mdn = TensorflowOnlineLearner.predict(self, state)
        # mdn = mdn[0]
        # print('prediction')
        # print(mdn)
        prediction = MixtureDensityNetwork.max_central_value(mixtures=np.squeeze(mdn[0]),
                                                             means=np.squeeze(mdn[1]),
                                                             variances=np.squeeze(mdn[2]))
        return prediction[0], np.sum(prediction[1])  # FIXME: Is this the best way to add the variances?

    def architecture(self):

        model = tf.map_fn(lambda frame: tf.image.resize_images(frame, (60, 80)), self.state_tensor)
        model = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), model)
        model = resnet_1(model)

        model = tf.layers.dense(model, units=64, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

        denoising_autoencoder = DenoisingAutoencoder(model, latent_size=24, noise=0.1, loss_function='cross_entropy')
        #
        if not self.training:
            self.vector_field = tf.reduce_mean(denoising_autoencoder.decoder - model)
            #     self.log_density = tf.reduce_mean((denoising_autoencoder.decoder - model) / 0.01)
            #     self.hessian = tf.reduce_mean(tf.subtract(tf.gradients(denoising_autoencoder.decoder, model), 1.))
            self.fortified_loss = denoising_autoencoder.loss
        else:
            self.fortified_loss = denoising_autoencoder.loss

        model = tf.layers.dense(denoising_autoencoder.decoder, units=32, activation=tf.nn.tanh,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

        loss, components, _ = MixtureDensityNetwork.create(model, self.action_tensor, number_mixtures=3)
        final_loss = loss + tf.reduce_mean(self.fortified_loss)
        tf.summary.scalar('total_loss', final_loss)
        return components, final_loss

    def get_optimizer(self, loss):
        return tf.train.AdagradOptimizer(1e-3).minimize(loss, global_step=self.global_step)

