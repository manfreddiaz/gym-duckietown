import tensorflow as tf
from learning_iil.learners.models.tf._layers import resnet_1, MixtureDensityNetwork

from ..tf_conditional_online_learner import GoalOrientedTensorflowOnlineLearner

tf.set_random_seed(1234)


class GoalOrientedLearnerOneResidualMDN(GoalOrientedTensorflowOnlineLearner):
    def __init__(self, world, x=0.0, y=0.0, theta=0.0, v=0.0):
        GoalOrientedTensorflowOnlineLearner.__init__(self, world, x=x, y=y, theta=theta, v=v)

    def predict(self, state, goal, horizon=1):
        mdn = GoalOrientedTensorflowOnlineLearner.predict(self, state, goal)
        return MixtureDensityNetwork.max_central_value(mdn[0], mdn[1], mdn[2])

    def architecture(self):
        model = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.state_tensor)
        model = resnet_1(model, keep_prob=1.0)
        model = tf.add(model, self.goal_tensor)
        model = tf.layers.dense(model, units=512, activation=tf.nn.tanh,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
        model = tf.layers.dense(model, units=64, activation=tf.nn.tanh,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))

        loss, mdn = MixtureDensityNetwork.create(model, self.action_tensor, number_mixtures=5)

        return mdn, loss

    def get_optimizer(self, loss):
        return tf.train.AdagradOptimizer(1e-3).minimize(loss, global_step=self.global_step)

