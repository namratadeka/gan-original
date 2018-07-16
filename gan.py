import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten


class Gan(object):
    '''

    '''
    def __init__(self):
        '''

        '''

    def generator(self):
        '''

        :return:
        '''

    def discriminator(self, x):
        '''

        :return:
        '''
        mu = 0
        sigma = 0.1

        conv1_W = tf.Variable(tf.truncated_normal(shape=(8,8,1,32), mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros(32))
        conv1 = tf.nn.conv2d(x, conv1_W, strides=[1,1,1,1], padding='VALID') + conv1_b

        conv1 = tf.nn.max_pool(conv1, ksize=[1,4,4,1], strides=[1,2,2,1], padding='VALID')

        conv2_W = tf.Variable(tf.truncated_normal(shape=(8, 8, 1, 32), mean=mu, stddev=sigma))
        conv2_b = tf.Variable(tf.zeros(32))
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

        conv2 = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 192), mean=mu, stddev=sigma))
        conv3_b = tf.Variable(tf.zeros(192))
        conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b

        conv3 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        fc0 = flatten(conv3)

        fc1_W = tf.Variable(tf.truncated_normal(shape=(,500), mean=mu, stddev=sigma))
