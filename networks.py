import tensorflow as tf


def xavier_init(size):
    '''

    :param size:
    :return:
    '''
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


class Network(object):
    '''

    '''
    def generator_layers_mnist(self):
        '''

        :return:
        '''
        self.g_fc1_W = tf.Variable(xavier_init([100, 128]))
        self.g_fc1_b = tf.Variable(tf.zeros(128))

        self.g_fc2_W = tf.Variable(xavier_init([128, 784]))
        self.g_fc2_b = tf.Variable(tf.zeros(784))

        self.theta_G = [self.g_fc1_W, self.g_fc1_W, self.g_fc2_W, self.g_fc2_b]

    def generator_ops_mnist(self,x):
        '''

        :param x:
        :return:
        '''
        fc1 = tf.matmul(x, self.g_fc1_W) + self.g_fc1_b
        fc1 = tf.nn.relu(fc1)
        fc3 = tf.matmul(fc1, self.g_fc2_W) + self.g_fc2_b
        fc3 = tf.nn.tanh(fc3)
        return fc3

    def discriminator_layers_mnist(self):
        '''

        :param x:
        :return:
        '''
        self.d_fc1_W = tf.Variable(xavier_init([784, 128]))
        self.d_fc1_b = tf.Variable(tf.zeros(128))

        self.d_fc2_W = tf.Variable(xavier_init([128, 1]))
        self.d_fc2_b = tf.Variable(tf.zeros(1))

        self.theta_D = [self.d_fc1_W, self.d_fc1_b, self.d_fc2_W, self.d_fc2_b]

    def discriminator_ops_mnist(self, x):
        '''

        :return:
        '''
        fc1 = tf.matmul(x, self.d_fc1_W) + self.d_fc1_b
        fc1 = tf.nn.relu(fc1)
        logit = tf.matmul(fc1, self.d_fc2_W) + self.d_fc2_b
        prob = tf.nn.sigmoid(logit)
        return prob, logit

    def generator_layers_cifar10(self):
        '''

        :return:
        '''
        self.g_fc1_W = tf.Variable(xavier_init([3072, 8000]))
        self.g_fc1_b = tf.Variable(tf.zeros(8000))

        self.g_fc2_W = tf.Variable(xavier_init([8000, 8000]))
        self.g_fc2_b = tf.Variable(tf.zeros(8000))

        self.theta_G = [self.g_fc1_W, self.g_fc1_W, self.g_fc2_W, self.g_fc2_b]

    def generator_ops_cifar10(self,x):
        '''

        :param x:
        :return:
        '''
        fc1 = tf.matmul(x, self.g_fc1_W) + self.g_fc1_b
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.matmul(fc1, self.g_fc2_W) + self.g_fc2_b
        fc2 = tf.nn.tanh(fc2)
        conv0 = tf.reshape(fc2, shape=[-1, 10, 10, 80])
        self.deconv = tf.layers.conv2d_transpose(conv0, filters=3, kernel_size=(5, 5), strides=(3, 3))

        # self.theta_G.append(self.deconv)
        return self.deconv

    def discriminator_layers_cifar10(self):
        '''

        :return:
        '''
        mu = 0
        sigma = 0.01
        self.d_conv1_W = tf.Variable(tf.truncated_normal(shape=(8,8,3,32), mean=mu, stddev=sigma))
        self.d_conv1_b = tf.Variable(tf.zeros(32))

        self.d_conv2_W = tf.Variable(tf.truncated_normal(shape=(8, 8, 32, 32), mean=mu, stddev=sigma))
        self.d_conv2_b = tf.Variable(tf.zeros(32))

        self.d_conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 192), mean=mu, stddev=sigma))
        self.d_conv3_b = tf.Variable(tf.zeros(192))

        self.d_fc1_W = tf.Variable(xavier_init([3072, 500]))
        self.d_fc1_b = tf.Variable(tf.zeros(500))

        self.d_fc2_W = tf.Variable(xavier_init([5, 1]))
        self.d_fc2_b = tf.Variable(tf.zeros(1))

        self.theta_D = [self.d_conv1_W, self.d_conv1_b, self.d_conv2_W, self.d_conv2_b, self.d_conv3_W, self.d_conv3_b,
                        self.d_fc1_W, self.d_fc1_b, self.d_fc2_W, self.d_fc2_b]

    def discriminator_ops_cifar10(self, x):
        '''

        :param x:
        :return:
        '''
        padding_1 = tf.constant([[0, 0], [4, 4], [4, 4], [0, 0]])
        x = tf.pad(x, padding_1, "CONSTANT")
        conv1 = tf.nn.conv2d(x, self.d_conv1_W, strides=[1, 1, 1, 1], padding='VALID') + self.d_conv1_b
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID')

        padding_2 = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
        conv1 = tf.pad(conv1, padding_2, "CONSTANT")
        conv2 = tf.nn.conv2d(conv1, self.d_conv2_W, strides=[1, 1, 1, 1], padding='VALID') + self.d_conv2_b
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID')

        padding_3 = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
        conv2 = tf.pad(conv2, padding_3, "CONSTANT")
        conv3 = tf.nn.conv2d(conv2, self.d_conv3_W, strides=[1, 1, 1, 1], padding='VALID') + self.d_conv3_b
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        fc0 = tf.contrib.layers.flatten(conv3)

        fc1 = tf.matmul(fc0, self.d_fc1_W) + self.d_fc1_b
        fc1 = tf.contrib.layers.maxout(fc1, num_units=5)

        logit = tf.matmul(fc1, self.d_fc2_W) + self.d_fc2_b
        prob = tf.nn.sigmoid(logit)
        return prob, logit