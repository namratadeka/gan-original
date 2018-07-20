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
