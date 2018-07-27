import tensorflow as tf
from learning_iil.learners.models.tf.tf_online_learner import TensorflowOnlineLearner

from .._layers import resnet_1

tf.set_random_seed(1234)


class ResnetOneRegression(TensorflowOnlineLearner):
    def explore(self, state, horizon=1):
        pass

    def __init__(self, name=None):
        self.name = name
        TensorflowOnlineLearner.__init__(self)

    def predict(self, state, horizon=1):
        regression = TensorflowOnlineLearner.predict(self, state)
        return regression, 0.0, 0.0

    def architecture(self):
        model = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.state_tensor)
        model = resnet_1(model)
        model = tf.layers.dense(model, units=64, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
        model = tf.layers.dense(model, self.action_tensor.shape[1])
        with tf.name_scope('losses'):
            loss = tf.losses.mean_squared_error(model, self.action_tensor)
            tf.summary.scalar('mse', loss)

        return [model], loss

    def get_optimizer(self, loss):
        return tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=self.global_step)

