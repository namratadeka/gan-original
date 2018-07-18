import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import pdb

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

        padding_1 = tf.constant([[0,0], [4, 4], [4, 4], [0,0]])
        x = tf.pad(x, padding_1, "CONSTANT")

        conv1_W = tf.Variable(tf.truncated_normal(shape=(8,8,3,32), mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros(32))
        conv1 = tf.nn.conv2d(x, conv1_W, strides=[1,1,1,1], padding='VALID') + conv1_b

        conv1 = tf.nn.max_pool(conv1, ksize=[1,4,4,1], strides=[1,2,2,1], padding='VALID')

        padding_2 = tf.constant([[0,0], [3,3], [3,3], [0,0]])
        conv1 = tf.pad(conv1, padding_2, "CONSTANT")

        conv2_W = tf.Variable(tf.truncated_normal(shape=(8, 8, 32, 32), mean=mu, stddev=sigma))
        conv2_b = tf.Variable(tf.zeros(32))
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

        conv2 = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID')

        padding_3 = tf.constant([[0,0], [3,3], [3,3], [0,0]])
        conv2 = tf.pad(conv2, padding_3, "CONSTANT")

        conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 192), mean=mu, stddev=sigma))
        conv3_b = tf.Variable(tf.zeros(192))
        conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b

        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        fc0 = flatten(conv3)

        fc1_W = tf.Variable(tf.truncated_normal(shape=(3072,500), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(500))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

        fc1 = tf.contrib.layers.maxout(fc1, num_units=5)

        fc2_W = tf.Variable(tf.truncated_normal(shape=(5,1), mean=mu, stddev=sigma))
        fc2_b = tf.Variable(tf.zeros(1))
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b

        logits = tf.nn.sigmoid(fc2)
        return logits


g = Gan()
img = np.random.random_sample((1,32,32,3))
x = tf.placeholder(tf.float32, (None, 32,32,3))
logits = g.discriminator(x)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(logits, feed_dict = {x: img})
    print(res)