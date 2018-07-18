import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import pdb

class Gan(object):
    '''

    '''
    def __init__(self):
        '''

        '''
        self.epochs = 10
        self.d_epochs = 5
        self.minibatch = 128
        self.data = self.get_data()
        self.train_ops()

    def train_ops(self):
        '''

        :return:
        '''
        self.noise = tf.placeholder(tf.float32, (None, 3072))
        self.image = tf.placeholder(tf.float32, (None, 32, 32, 3))
        self.d_loss = self.discriminator_loss(self.noise, self.image)
        self.g_loss = self.generator_loss(self.noise)
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.004, momentum=0.5)
        self.train_generator = optimizer.minimize(self.g_loss)
        self.train_discriminator = optimizer.minimize(-1*self.d_loss)
        self.generated_img = self.generator(self.noise)
        self.saver = tf.train.Saver()

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epochs):
                for j in range(self.d_epochs):
                    X_noise = self.noise_generator(self.minibatch)
                    X_data = self.data[np.random.choice(self.data.shape[0], self.minibatch)]
                    sess.run(self.train_discriminator, feed_dict={self.noise: X_noise, self.image: X_data})
                    print("discriminator loss: %f" % (sess.run(self.d_loss, feed_dict={self.noise: X_noise, self.image: X_data})))

                X_noise = self.noise_generator(self.minibatch)
                sess.run(self.train_generator, feed_dict={self.noise: X_noise})
                print("EPOCH %d"%i)
                print("generator loss: %f"%(sess.run(self.g_loss, feed_dict={self.noise: X_noise})))
            self.saver.save(sess, './gan')
            print('Model saved.')

    def test(self):
        '''

        :return:
        '''
        with tf.Session() as sess:
            x = self.noise_generator(1)
            self.saver.restore(sess, tf.train.latest_checkpoint('.'))
            img = sess.run(self.generated_img, feed_dict={self.noise: x})
            plt.imshow(img[0]*255)
            plt.show()

    def discriminator_loss(self, noise, image):
        '''

        :param noise:
        :param image:
        :return:
        '''
        gen_image = self.generator(noise)
        gen_image_prob = self.discriminator(gen_image)
        image_prob = self.discriminator(image)
        return tf.reduce_mean(tf.log(image_prob) + tf.log(1 - gen_image_prob))

    def generator_loss(self, noise):
        '''

        :param noise:
        :return:
        '''
        gen_image = self.generator(noise)
        gen_image_prob = self.discriminator(gen_image)
        return tf.reduce_mean(tf.log(1 - gen_image_prob))

    def generator(self, x):
        '''

        :return:
        '''
        mu = 0.5
        sigma = 0.01

        fc1_W = tf.Variable(tf.truncated_normal(shape=(3072,8000), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(8000))
        fc1 = tf.matmul(x, fc1_W) + fc1_b

        fc1 = tf.nn.relu(fc1)

        fc2_W = tf.Variable(tf.truncated_normal(shape=(8000,8000), mean=mu, stddev=sigma))
        fc2_b = tf.Variable(tf.zeros(8000))
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b

        fc2 = tf.nn.sigmoid(fc2)

        conv0 = tf.reshape(fc2, shape=np.array([-1,10,10,80]))

        deconv = tf.layers.conv2d_transpose(conv0, filters=3, kernel_size=(5,5), strides=(3,3))
        return deconv

    def discriminator(self, x):
        '''

        :return:
        '''
        mu = 0.5
        sigma = 0.01

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

        fc0 = tf.contrib.layers.flatten(conv3)

        fc1_W = tf.Variable(tf.truncated_normal(shape=(3072,500), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(500))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b

        fc1 = tf.contrib.layers.maxout(fc1, num_units=5)

        fc2_W = tf.Variable(tf.truncated_normal(shape=(5,1), mean=mu, stddev=sigma))
        fc2_b = tf.Variable(tf.zeros(1))
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b

        logits = tf.nn.sigmoid(fc2)
        return logits

    def noise_generator(self, m):
        '''

        :param m:
        :return:
        '''
        return np.random.normal(size=(m,3072))

    def get_data(self):
        '''

        :return:
        '''
        source = '../data/cifar-10-batches-py/'
        files = ["%s//%s"%(source,f) for f in os.listdir(source) if not (f.endswith('.html') or f.endswith('.meta'))]
        r = np.empty((0, 32, 32), dtype=np.uint8)
        g = np.empty((0, 32, 32), dtype=np.uint8)
        b = np.empty((0, 32, 32), dtype=np.uint8)
        imgs = []
        for file in files:
            dict = pickle.load(open(file,'rb'), encoding='bytes')
            idx = np.where(np.array(dict[b'labels']) == 5)[0]
            r = np.append(r, (dict[b'data'][idx][:,:1024]).reshape((len(idx), 32, 32)), axis=0)
            g = np.append(g, (dict[b'data'][idx][:,1024:2048]).reshape((len(idx), 32, 32)), axis=0)
            b = np.append(b, (dict[b'data'][idx][:,2048:3072]).reshape((len(idx), 32, 32)), axis=0)
        for i in range(r.shape[0]):
            img = np.dstack((r[i,:], g[i,:], b[i,:]))
            imgs.append(img)
        return np.array(imgs)/255.0


g = Gan()
g.train()
g.test()