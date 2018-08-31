import numpy as np
import tensorflow as tf

from ..tf_parametrization import TensorflowParametrization
from .._layers import resnet_1, MixtureDensityNetwork


class ResnetOneMixture(TensorflowParametrization):
    def __init__(self):
        TensorflowParametrization.__init__(self)

    def test(self, state, horizon=1):
        mdn = TensorflowParametrization.test(self, state)
        mdn = mdn[0]
        prediction = MixtureDensityNetwork.max_central_value(mixtures=np.squeeze(mdn[0]),
                                                             means=np.squeeze(mdn[1]),
                                                             variances=np.squeeze(mdn[2]))
        return prediction[0]

    def architecture(self):
        model = resnet_1(self.state_tensor, keep_prob=0.5 if self.training else 1.0)
        model = tf.layers.dense(model, units=64, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
        model = tf.layers.dense(model, units=32, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

        loss, components, _ = MixtureDensityNetwork.create(model, self.action_tensor, number_mixtures=3)
        return components, loss

