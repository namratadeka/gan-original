import numpy as np
import tensorflow as tf

import utils
from networks import Network


class GAN(object):
    def __init__(self):
        '''

        '''
        self.BATCH_SIZE = 32
        self.EPOCHS = 100000
        # self.image_dim = [-1, 32, 32, 3]                          # CIFAR-10
        self.image_dim = [-1, 28, 28, 1]                           # MNIST
        # self.discriminator_input_dim = (None, 32, 32, 3)          # CIFAR-10
        self.discriminator_input_dim = (None, 784)                 # MNIST
        self.net = Network()
        # self.data = utils.get_cifar10()                           # CIFAR-10
        self.data = utils.get_mnist()                              # MNIST
        # self.logdir = "train_logs/cifar10/"                   # CIFAR-10
        self.logdir = "train_logs/mnist/"                      # MNIST

    def train_ops(self):
        '''

        :return:
        '''
        self.input_placeholder = tf.placeholder(tf.float32,
                                           shape=self.discriminator_input_dim,
                                           name='input')
        random_z = tf.random_normal([self.BATCH_SIZE, 100], mean=0.0, stddev=1.0, name='random_z')

        self.generator = self.net.mnist_generator(random_z, is_training=True, name='generator')

        self.real_discriminator = self.net.mnist_discriminator(self.input_placeholder, is_training=True, name='discriminator')
        self.fake_discriminator = self.net.mnist_discriminator(self.generator, is_training=True, reuse=True, name='discriminator')

        self.real_discriminator_loss = tf.losses.sigmoid_cross_entropy(tf.constant(1, shape=[self.BATCH_SIZE]),
                                                                  self.real_discriminator,
                                                                  scope='real_discriminator_loss')
        self.fake_discriminator_loss = tf.losses.sigmoid_cross_entropy(tf.constant(0, shape=[self.BATCH_SIZE]),
                                                                  self.fake_discriminator,
                                                                  scope='fake_discriminator_loss')
        self.discriminator_loss = self.real_discriminator_loss + self.fake_discriminator_loss

        self.generator_loss = tf.losses.sigmoid_cross_entropy(tf.constant(1, shape=[self.BATCH_SIZE]),
                                                         self.fake_discriminator,
                                                         scope='generator_loss')
        ## training variables
        self.discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                    scope='discriminator')
        self.generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                scope='generator')
        ## update ops
        discriminator_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                     scope='discriminator')
        generator_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                 scope='generator')

        global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.control_dependencies(discriminator_update_ops):
            self.train_discriminator = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5). \
                minimize(self.discriminator_loss, var_list=self.discriminator_variables)
        with tf.control_dependencies(generator_update_ops):
            self.train_generator = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5). \
                minimize(self.generator_loss, var_list=self.generator_variables, global_step=global_step)

    def summary_ops(self):
        '''

        :return:
        '''
        self.summary_discriminator = tf.summary.merge([
            tf.summary.scalar('summary/real_discriminator_loss', self.real_discriminator_loss),
            tf.summary.scalar('summary/fake_discriminator_loss', self.fake_discriminator_loss),
            tf.summary.scalar('summary/discriminator_loss', self.discriminator_loss)])

        input_img = tf.reshape(self.input_placeholder, self.image_dim)
        gen_img = tf.reshape(self.generator, self.image_dim)
        # input_visualisation = tf.cast(((self.input_placeholder / 2.0) + 0.5) * 255.0, tf.uint8)
        # generator_visualisation = tf.cast(((self.generator / 2.0) + 0.5) * 255.0, tf.uint8)
        input_visualisation = tf.cast(((input_img / 2.0) + 0.5) * 255.0, tf.uint8)
        generator_visualisation = tf.cast(((gen_img / 2.0) + 0.5) * 255.0, tf.uint8)

        self.summary_input = tf.summary.image('summary/input',
                                         input_visualisation, max_outputs=3)

        self.summary_generator = tf.summary.merge([
            tf.summary.image('summary/generator', generator_visualisation, max_outputs=3),
            tf.summary.scalar('summary/generator_loss', self.generator_loss)])

    def train(self):
        '''

        :return:
        '''
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            logwriter = tf.summary.FileWriter(self.logdir, sess.graph)
            for i in range(self.EPOCHS):
                inp = self.data[np.random.choice(self.data.shape[0], self.BATCH_SIZE)]

                (_, sum_dis) = sess.run((self.train_discriminator, self.summary_discriminator),
                                        feed_dict={self.input_placeholder: inp})

                (_, sum_gen) = sess.run((self.train_generator, self.summary_generator))

                s = sess.run(self.summary_input, feed_dict={self.input_placeholder: inp})

                if i % 100 == 0:
                    print(i)
                    logwriter.add_summary(sum_dis, i)
                    logwriter.add_summary(sum_gen, i)
                    logwriter.add_summary(s, i)


gan = GAN()
gan.train_ops()
gan.summary_ops()
gan.train()
