import tensorflow as tf


def residual_block(x, size, dropout=False, dropout_prob=0.5):
    residual = tf.layers.batch_normalization(x)  # TODO: check if the defaults in Tf are the same as in Keras
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(residual, filters=size, kernel_size=3, strides=2, padding='same')
    residual = tf.layers.batch_normalization(residual)
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(residual, filters=size, kernel_size=3, padding='same')
    return residual


def model(input_tensor):

    # nn = tf.map_fn(lambda frame: tf.image.resize_images(frame, (60, 80)), input_tensor)
    # nn = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), nn)

    nn = tf.layers.conv2d(input_tensor, filters=32, kernel_size=5, strides=2, padding='same')
    nn = tf.layers.max_pooling2d(nn, pool_size=3, strides=2)

    # residual = tf.layers.batch_normalization(nn)  # TODO: check if the defaults in Tf are the same as in Keras
    residual = tf.nn.relu(nn)
    residual = tf.layers.conv2d(residual, filters=32, kernel_size=3, strides=2, padding='same')
    # residual = tf.layers.batch_normalization(residual)
    residual = tf.nn.relu(residual)
    residual = tf.layers.conv2d(residual, filters=32, kernel_size=3, padding='same')

    nn = tf.layers.conv2d(nn, filters=32, kernel_size=1, strides=2, padding='same')
    nn = tf.keras.layers.add([residual, nn])

    # TODO: check https://github.com/raghakot/keras-resnet for the absence of RELU after merging

    # nn = tf.layers.batch_normalization(nn)
    # nn = tf.nn.relu(nn)
    # nn = tf.keras.layers.GlobalAveragePooling2D()(nn)
    nn = tf.layers.flatten(nn)

    nn = tf.layers.dense(nn, units=64, activation=tf.nn.relu)
    # model = tf.nn.dropout(model, keep_prob=0.5)
    nn = tf.layers.dense(nn, units=32, activation=tf.nn.relu)

    nn = tf.layers.dense(nn, 2)

    return nn
