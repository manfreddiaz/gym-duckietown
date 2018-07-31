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
        self.fortified_loss = NotImplementedError

    def predict(self, state, horizon=1):
        mdn = TensorflowOnlineLearner.predict(self, state)
        # print('prediction')
        # print(mdn)
        prediction = MixtureDensityNetwork.max_maximum_mixture(mixtures=np.squeeze(mdn[0]),
                                                             means=np.squeeze(mdn[1]),
                                                             variances=np.squeeze(mdn[2]))
        return prediction[0], np.sum(prediction[1])  # FIXME: Is this the best way to add the variances?

    def architecture(self):
        model = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.state_tensor)
        model = resnet_1(model, keep_prob=1.0)
        # TRY: tanh + 64 or change to (relu, crelu), without dense
        # model = tf.layers.dense(model, units=512, activation=tf.nn.relu,
        #                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
        #                         bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
        #                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
        # model = tf.layers.dense(model, units=100, activation=tf.nn.relu,
        #                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
        #                         bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
        #                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
        model = tf.layers.dense(model, units=64, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

        fortified_layer = DenoisingAutoencoder(model)
        self.fortified_loss = fortified_layer.loss

        loss, components, _ = MixtureDensityNetwork.create(model, self.action_tensor, number_mixtures=3)
        return components, loss + 1e-2 * self.fortified_loss

    def get_optimizer(self, loss):
        return tf.train.AdagradOptimizer(1e-3).minimize(loss, global_step=self.global_step)

