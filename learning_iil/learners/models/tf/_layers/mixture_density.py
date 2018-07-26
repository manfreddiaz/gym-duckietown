from operator import itemgetter
import tensorflow as tf
import numpy as np

tf.set_random_seed(1234)

L2_LAMBDA = 0.01


def loss_1(y, mean, variance, mixtures):
    diff = y - mean
    squared_diff = -tf.square(diff / variance) / 2.0
    pdf = tf.log(mixtures) + squared_diff - 0.5 * tf.log(np.pi * 2.0 * tf.square(variance))
    log_sum_exp = -tf.reduce_logsumexp(pdf, axis=1)
    return tf.reduce_mean(log_sum_exp)


def loss(y, mixtures, means, variances):
    diff = y - means
    log_likelihood = tf.log(mixtures) - 0.5 * tf.log(2.0 * np.pi * tf.square(variances)) - 0.5 * tf.square(diff * tf.reciprocal(variances))
    log_sum_exp = -tf.reduce_logsumexp(log_likelihood, axis=1)
    final_loss = tf.reduce_mean(log_sum_exp)
    tf.summary.scalar('loss', final_loss)
    return final_loss


class MixtureDensityNetwork:
    @staticmethod
    def create(input_layer, output_layer, number_mixtures,
               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
               bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01)):
        output_dim = output_layer.get_shape().as_list()[0]

        layer = tf.layers.dense(input_layer,
                                units=3 * number_mixtures * output_dim,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                kernel_regularizer=kernel_regularizer)
        mixtures_activations, means, variance_activations = tf.split(layer, 3, axis=1)

        mixtures = tf.nn.softmax(mixtures_activations, name='mixtures_component')
        variances = tf.exp(variance_activations, name='variances_component')

        i_mixtures = tf.split(mixtures, number_mixtures, axis=1)
        i_means = tf.split(means, number_mixtures, axis=1)
        i_variances = tf.split(variances, number_mixtures, axis=1)

        for i in range(number_mixtures):
            with tf.name_scope('mixture_{}'.format(i)):
                tf.summary.scalar('mixture', tf.reshape(i_mixtures[i], shape=[]))
                tf.summary.scalar('mean', tf.reshape(i_means[i], shape=[]))
                tf.summary.scalar('variance', tf.reshape(i_variances[i], shape=[]))

        conditional_average = tf.reduce_sum(tf.multiply(means, mixtures))
        with tf.name_scope('statistisc'):
            epistemic_total = tf.reduce_sum(tf.multiply(mixtures, variances))
            tf.summary.scalar('epistemic', epistemic_total)
            aleatoric_total = tf.reduce_sum(tf.multiply(mixtures, tf.square(np.subtract(means, conditional_average))))
            tf.summary.scalar('aleatoric', aleatoric_total)

        return loss(output_layer, mixtures, means, variances), [mixtures, means, variances], layer

    @staticmethod
    def conditional_average(mixtures, means, std_dev):
        conditional_average = np.sum(np.multiply(means, mixtures))
        epistemic = np.sum(np.multiply(mixtures, std_dev))
        aleatoric = np.sum(np.multiply(mixtures, np.square(np.subtract(means, conditional_average))))
        return conditional_average, aleatoric, epistemic

    @staticmethod
    def max_maximum_mixture(mixtures, means, variances):
        conditional_average = np.sum(np.multiply(means, mixtures))
        max_mixture = np.argmax(mixtures)
        mean = means[max_mixture]
        aleatoric = variances[max_mixture]
        epistemic = (mean - conditional_average) ** 2
        return mean, aleatoric, epistemic, mixtures[max_mixture], max_mixture

    @staticmethod
    def max_central_value(mixtures, means, std_dev):
        conditional_average = np.sum(np.multiply(means, mixtures))
        central_value = np.divide(mixtures, std_dev)

        epistemic_total = np.sum(np.multiply(mixtures, std_dev))
        aleatoric_total = np.sum(np.multiply(mixtures, np.square(np.subtract(means, conditional_average))))

        max_mixture = np.argmax(central_value)

        mean = means[max_mixture]

        print(epistemic_total, aleatoric_total, epistemic_total + aleatoric_total)

        # TODO: Find a function to make this bigger
        aleatoric = std_dev[max_mixture] # ** 2
        epistemic = (mean - conditional_average) ** 2

        return mean, aleatoric_total + epistemic_total, epistemic, mixtures[max_mixture], max_mixture

    @staticmethod
    def top_k_mixtures(k, mixtures, means, variances):
        top_k = np.argpartition(mixtures, -k)[-k:]
        top_k_means = means[top_k]
        conditional_average = np.sum(np.multiply(means, mixtures))
        top_k_epistemic = np.square(top_k_means - conditional_average)
        top_k_components = zip(mixtures[top_k], means[top_k], variances[top_k], top_k_epistemic, top_k)
        top_k_components = list(zip(*sorted(top_k_components, reverse=True, key=itemgetter(0))))
        return top_k_components[1], top_k_components[2], top_k_components[3], top_k_components[0], top_k_components[
            4]

    @staticmethod
    def top_k_central_value(k, mixtures, means, variances):
        central_value = np.array(list(map(lambda x, y: x / y, mixtures, variances)))
        top_k = np.argpartition(central_value, -k)[-k:]
        top_k_means = means[top_k]
        conditional_average = np.sum(np.multiply(means, mixtures))
        top_k_epistemic = np.square(top_k_means - conditional_average)
        tup = zip(mixtures[top_k], means[top_k], variances[top_k], top_k_epistemic)
        tup = tuple(zip(*sorted(tup, reverse=True)))
        return tup[1], tup[2], tup[3]
