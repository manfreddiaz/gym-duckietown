import tensorflow as tf

# optimization
LEARNING_RATES = [1e-3, 1e-4, 1e-2, 1e-1]

weight_decay = 1e-4
OPTIMIZATION_METHODS_NAMES = ['adamw', 'adam', 'adagrad', 'rmsprop', 'ggt', 'sgd_wr']

def adamw(learning_rate, weight_decay):
    return tf.contrib.opt.AdamWOptimizer(
        weight_decay=weight_decay,
        learning_rate=learning_rate
    )

def adam(learning_rate):
    return tf.train.AdamOptimizer(
        learning_rate=learning_rate
    )

def adagrad(learning_rate):
    return tf.train.AdagradOptimizer(
        learning_rate=learning_rate
    )

def rmsprop(learning_rate):
    return tf.train.RMSPropOptimizer(
        learning_rate=learning_rate
    )

def ggt(learning_rate):
    return tf.contrib.opt.GGTOptimizer(
        learning_rate=learning_rate
    )

def sgd_wr(learning_rate, global_step, first_decay_steps):
    learning_rate_warm_restarts = tf.train.cosine_decay_restarts(
        learning_rate=learning_rate,
        global_step=global_step,
        first_decay_steps=first_decay_steps,
        name='lr_wr'
    )

    # logging
    tf.summary.scalar('lr_wr', learning_rate_warm_restarts)

    return tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate_warm_restarts
    )


def optimizer(optimizer_iteration, learning_rate_iteration, parametrization, task_metadata):
    if optimizer_iteration == 0:
        return adamw(learning_rate=LEARNING_RATES[learning_rate_iteration], weight_decay=weight_decay)
    elif optimizer_iteration == 1:
        return adam(learning_rate=LEARNING_RATES[learning_rate_iteration])
    elif optimizer_iteration == 2:
        return adagrad(learning_rate=LEARNING_RATES[learning_rate_iteration])
    elif optimizer_iteration == 3:
        return rmsprop(learning_rate=LEARNING_RATES[learning_rate_iteration])
    elif optimizer_iteration == 4:
        return
    elif optimizer_iteration == 5:
        return sgd_wr(
            learning_rate=LEARNING_RATES[learning_rate_iteration],
            global_step=parametrization.global_step,
            first_decay_steps=task_metadata[0] * task_metadata[2]
        )
    else:
        raise IndexError()