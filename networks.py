import tensorflow as tf


def xavier_std(size):
    '''

    :param size:
    :return:
    '''
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return xavier_stddev


class Network(object):
    '''

    '''

    def cifar10_generator(self, inputs, is_training=False, reuse=False, name='generator'):
        '''

        :param is_training:
        :param reuse:
        :param name:
        :return:
        '''
        with tf.variable_scope(name, reuse=reuse):
            normal_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

            inputs = tf.expand_dims(tf.expand_dims(inputs, 1), 1)

            ## deconv output = [batch, 4, 4, 1024]
            net = tf.layers.conv2d_transpose(inputs,
                                             filters=1024,
                                             kernel_size=4,
                                             padding='valid',
                                             kernel_initializer=normal_initializer,
                                             trainable=is_training,
                                             name='tconv1')
            net = tf.layers.batch_normalization(net,
                                                training=is_training,
                                                name='tconv1/batch_normalization')
            net = tf.nn.relu(net, name='tconv1/relu')
            # pdb.set_trace()
            ##deconv output = [batch, 8, 8, 256]
            net = tf.layers.conv2d_transpose(net,
                                             filters=256,
                                             kernel_size=4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=normal_initializer,
                                             trainable=is_training,
                                             name='tconv2')
            net = tf.layers.batch_normalization(net,
                                                training=is_training,
                                                name='tconv2/batch_normalization')
            net = tf.nn.relu(net, name='tconv2/relu')

            # deconv output = [batch, 16, 16, 64]
            net = tf.layers.conv2d_transpose(net,
                                             filters=64,
                                             kernel_size=4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=normal_initializer,
                                             trainable=is_training,
                                             name='tconv3')

            net = tf.layers.batch_normalization(net,
                                                training=is_training,
                                                name='tconv3/batch_normalization')
            net = tf.nn.relu(net, name='tconv3/relu')

            # deconv output = [batch, 32, 32, 3]
            net = tf.layers.conv2d_transpose(net,
                                             filters=3,
                                             kernel_size=4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=normal_initializer,
                                             trainable=is_training,
                                             name='tconv4')

            net = tf.tanh(net, name='tconv4/tanh')

            return net

    def cifar10_discriminator(self, inputs, is_training=False, reuse=False, name='discriminator'):
        '''

        :param is_training:
        :param reuse:
        :param name:
        :return:
        '''
        with tf.variable_scope(name, reuse=reuse):
            normal_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

            # Convolution outputs [batch, 16, 16, 64]
            net = tf.layers.conv2d(inputs,
                                   filters=64,
                                   kernel_size=4,
                                   strides=2,
                                   padding='same',
                                   kernel_initializer=normal_initializer,
                                   trainable=is_training,
                                   name='conv1')

            net = tf.nn.leaky_relu(net, alpha=0.2, name='conv1/leaky_relu')

            # Convolution outputs [batch, 8, 8, 256]
            net = tf.layers.conv2d(net,
                                   filters=256,
                                   kernel_size=4,
                                   strides=2,
                                   padding='same',
                                   kernel_initializer=normal_initializer,
                                   trainable=is_training,
                                   name='conv2')

            net = tf.layers.batch_normalization(net,
                                                training=is_training,
                                                name='conv2/batch_normalization')

            net = tf.nn.leaky_relu(net, alpha=0.2, name='conv2/leaky_relu')

            # Convolution outputs [batch, 4, 4, 1024]
            net = tf.layers.conv2d(net,
                                   filters=1024,
                                   kernel_size=4,
                                   strides=2,
                                   padding='same',
                                   kernel_initializer=normal_initializer,
                                   trainable=is_training,
                                   name='conv3')

            net = tf.layers.batch_normalization(net,
                                                training=is_training,
                                                name='conv3/batch_normalization')

            net = tf.nn.leaky_relu(net, alpha=0.2, name='conv3/leaky_relu')

            # Convolution outputs [batch, 1, 1, 1]
            net = tf.layers.conv2d(net,
                                   filters=1,
                                   kernel_size=4,
                                   padding='valid',
                                   kernel_initializer=normal_initializer,
                                   trainable=is_training,
                                   name='conv4')

            # Squeeze height and width dimensions
            net = tf.squeeze(net, [1, 2, 3])

            return net

    def mnist_generator(self, inputs, is_training=False, reuse=False, name='generator'):
        '''

        :param inputs:
        :param is_training:
        :param reuse:
        :param name:
        :return:
        '''
        with tf.variable_scope(name, reuse=reuse):
            normal_initializer = tf.random_normal_initializer(mean=0.0, stddev=xavier_std([100]))

            inputs = tf.expand_dims(tf.expand_dims(inputs, 1), 1)

            # FC layer 1: output dim = [batch, 128]
            net = tf.layers.dense(inputs,
                                  units=128,
                                  kernel_initializer=normal_initializer,
                                  trainable=is_training,
                                  name='fc1')
            # net = tf.layers.batch_normalization(net,
            #                                     training=is_training,
            #                                     name='fc1/batch_normalization')
            net = tf.nn.relu(net, name='fc1/relu')

            # FC layer 2: output dim = [batch, 784]
            net = tf.layers.dense(net,
                                  units=784,
                                  kernel_initializer=normal_initializer,
                                  trainable=is_training,
                                  name='fc2')
            # net = tf.layers.batch_normalization(net,
            #                                     training=is_training,
            #                                     name='fc2/batch_normalization')
            net = tf.nn.tanh(net, name='fc2/tanh')
            net = tf.reshape(net, [-1, 784])

            return net

    def mnist_discriminator(self, inputs, is_training=False, reuse=False, name='discriminator'):
        '''

        :param inputs:
        :param is_training:
        :param reuse:
        :param name:
        :return:
        '''
        with tf.variable_scope(name, reuse=reuse):
            normal_initializer = tf.random_normal_initializer(mean=0.0, stddev=xavier_std([100]))

            # FC layer 1: output dim = [batch, 128]
            net = tf.layers.dense(inputs,
                                  units=128,
                                  kernel_initializer=normal_initializer,
                                  trainable=is_training,
                                  name='fc1')
            net = tf.nn.relu(net, name='fc1/relu')

            # FC layer 2: output dim = [batch, 1]
            net = tf.layers.dense(net,
                                  units=1,
                                  kernel_initializer=normal_initializer,
                                  trainable=is_training,
                                  name='fc2')
            net = tf.nn.sigmoid(net, name='fc2/sigmoid')
            net = tf.squeeze(net, [1])

            return net
