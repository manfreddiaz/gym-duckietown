import numpy as np
import tensorflow as tf
from learning_iil.learners.models.tf.tf_online_learner import TensorflowOnlineLearner

from .._layers import resnet_1_dropout, MixtureDensityNetwork


class MonteCarloDropoutResnetOneMixture(TensorflowOnlineLearner):
    def explore(self, state, horizon=1):
        pass

    def __init__(self, mixtures=3):
        TensorflowOnlineLearner.__init__(self)
        self.mixtures = mixtures

    def predict(self, state, horizon=1):
        mdn = TensorflowOnlineLearner.predict(self, np.repeat(state, 16, axis=0))
        mdn = mdn[0]
        mixtures = np.mean(mdn[0], axis=0)
        means = np.mean(mdn[1], axis=0)
        variances = np.mean(mdn[2], axis=0)
        prediction = MixtureDensityNetwork.max_central_value(mixtures=np.squeeze(mixtures),
                                                             means=np.squeeze(means),
                                                             variances=np.squeeze(variances))
        return prediction[0], np.mean(prediction[1])  # FIXME: Is this the best way to add the variances?

    def architecture(self):
        model = tf.map_fn(lambda frame: tf.image.resize_images(frame, (60, 80)), self.state_tensor)
        model = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), model)

        model = resnet_1_dropout(model, keep_prob=0.5)
        model = tf.layers.dense(model, units=64, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        model = tf.nn.dropout(model, keep_prob=0.5)
        model = tf.layers.dense(model, units=32, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        model = tf.nn.dropout(model, keep_prob=0.5)

        loss, components, _ = MixtureDensityNetwork.create(model, self.action_tensor, number_mixtures=self.mixtures)

        return components, loss

    def get_optimizer(self, loss):
        return tf.train.AdagradOptimizer(1e-3).minimize(loss, global_step=self.global_step)

