import tensorflow as tf

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

def conv2d(inputs, filter_height, filter_width, output_channels, stride=(1, 1), padding='SAME', name='Conv2D'):
    input_channels = int(inputs.get_shape()[-1])
    fan_in = filter_height * filter_width * input_channels
    stddev = tf.sqrt(2.0 / fan_in)
    weights_shape = [filter_height, filter_width, input_channels, output_channels]
    biases_shape = [output_channels]

    with tf.variable_scope(name):
        filters_init = tf.truncated_normal_initializer(stddev=stddev)
        biases_init = tf.constant_initializer(0.1)

        filters = tf.get_variable(
            'weights', shape=weights_shape, initializer=filters_init, collections=['weights', 'variables'])
        biases = tf.get_variable(
            'biases', shape=biases_shape, initializer=biases_init, collections=['biases', 'variables'])
        # variable_summaries(filters , 'Weights')
        # variable_summaries(biases, 'Biases')
        return tf.nn.conv2d(inputs, filters, strides=[1, *stride, 1], padding=padding) + biases


def conv2d_nobias(inputs, filter_height, filter_width, output_chennels, stride=(1, 1), padding='SAME', name='Conv2D_nobias'):
    input_channels = int(inputs.get_shape()[-1])
    fan_in = filter_height * filter_width * input_channels
    stddev = tf.sqrt(2.0 / fan_in)
    weights_shape = [filter_height, filter_width, input_channels, output_chennels]

    with tf.variable_scope(name):
        filters_init = tf.truncated_normal_initializer(stddev=stddev)

        filters = tf.get_variable(
            'weights', shape=weights_shape, initializer=filters_init, collections=['weights', 'variables'])
        return tf.nn.conv2d(inputs, filters, strides=[1, *stride, 1], padding=padding)


def deconv2d(inputs, filter_height, filter_width, output_shape, stride=(1, 1), padding='SAME', name='Deconv2D'):
    input_channels = int(inputs.get_shape()[-1])
    output_channels = output_shape[-1]
    fan_in = filter_height * filter_width * output_channels
    stddev = tf.sqrt(2.0 / fan_in)
    weights_shape = [filter_height, filter_width, output_channels, input_channels]
    biases_shape = [output_channels]

    with tf.variable_scope(name):
        filters_init = tf.truncated_normal_initializer(stddev=stddev)
        biases_init = tf.constant_initializer(0.1)

        filters = tf.get_variable(
            'weights', shape=weights_shape, initializer=filters_init, collections=['weights', 'variables'])
        biases = tf.get_variable(
            'biases', shape=biases_shape, initializer=biases_init, collections=['biases', 'variables'])
        return tf.nn.conv2d_transpose(inputs, filters, output_shape, strides=[1, *stride, 1], padding=padding) + biases


def deconv2d_nobias(inputs, filter_height, filter_width, output_shape, stride=(1, 1), padding='SAME', name='Deconv2D_nobias'):
    input_channels = int(inputs.get_shape()[-1])
    output_channels = output_shape[-1]
    fan_in = filter_height * filter_width * output_channels
    stddev = tf.sqrt(2.0 / fan_in)
    weights_shape = [filter_height, filter_width, output_channels, input_channels]

    with tf.variable_scope(name):
        filters_init = tf.truncated_normal_initializer(stddev=stddev)

        filters = tf.get_variable(
            'weights', shape=weights_shape, initializer=filters_init, collections=['weights', 'variables'])
        return tf.nn.conv2d_transpose(inputs, filters, output_shape, strides=[1, *stride, 1], padding=padding)


def relu(inputs, name='Relu'):
    return tf.nn.relu(inputs, name)


def leaky_relu(inputs, leak=0.1, name='LeakyRelu'):
    with tf.name_scope(name):
        return tf.maximum(inputs, leak * inputs)


def batch_norm(inputs, decay, is_training, var_epsilon=1e-3, name='batch_norm'):
    with tf.variable_scope(name):
        scale = tf.Variable(tf.ones([int(inputs.get_shape()[-1])]))
        offset = tf.Variable(tf.zeros([int(inputs.get_shape()[-1])]))
        avg_mean = tf.Variable(tf.zeros([int(inputs.get_shape()[-1])]), trainable=False)
        avg_var = tf.Variable(tf.ones([int(inputs.get_shape()[-1])]), trainable=False)

        def get_batch_moments():
            batch_mean, batch_var = tf.nn.moments(inputs, list(range(len(inputs.get_shape()) - 1)))
            assign_mean = tf.assign(avg_mean, decay * avg_mean + (1.0 - decay) * batch_mean)
            assign_var = tf.assign(avg_var, decay * avg_var + (1.0 - decay) * batch_var)
            with tf.control_dependencies([assign_mean, assign_var]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        def get_avg_moments():
            return avg_mean, avg_var

        mean, var = tf.cond(is_training, get_batch_moments, get_avg_moments)
        return tf.nn.batch_normalization(inputs, mean, var, offset, scale, var_epsilon)
