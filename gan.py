import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mnist import MNIST


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


class Gan(object):
    '''

    '''
    def __init__(self):
        '''

        '''
        self.epochs = 10000
        self.d_epochs = 1
        self.minibatch = 128
        self.data = self.get_mnist()
        self.noise = tf.placeholder(tf.float32, (None, 100))
        self.image = tf.placeholder(tf.float32, (None, 784))
        self.discriminator_nodes(self.image)
        self.generator_nodes(self.noise)
        self.train_ops()
        self.gen_loss = []
        self.dis_loss = []
        self.prob = []

    def train_ops(self):
        '''

        :return:
        '''
        self.d_loss = self.discriminator_loss(self.noise, self.image)
        self.g_loss = self.generator_loss(self.noise)
        optimizer = tf.train.AdamOptimizer()
        self.generated_img = self.generator(self.noise)
        self.image_probability = self.expected_probability(self.noise)
        self.sample_probabilities, _ = self.discriminator(self.generated_img)

        ## MODIFIED LOSS FUNCTION
        # self.real_prob, self.real_logit = self.discriminator(self.image)
        # self.fake_prob, self.fake_logit = self.discriminator(self.generated_img)
        #
        # self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logit, labels=tf.ones_like(self.real_logit)))
        # self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit, labels=tf.zeros_like(self.fake_logit)))
        # self.d_loss = self.d_loss_real + self.d_loss_fake
        # self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit, labels=tf.ones_like(self.fake_logit)))

        self.train_generator = optimizer.minimize(self.g_loss, var_list=self.theta_G)
        self.train_discriminator = optimizer.minimize(self.d_loss, var_list=self.theta_D)

        self.saver = tf.train.Saver()

    def train(self):
        '''

        :return:
        '''
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epochs):
                print("EPOCH %d" % i)
                for j in range(self.d_epochs):
                    X_noise = self.noise_generator(self.minibatch)
                    X_data = self.data[np.random.choice(self.data.shape[0], self.minibatch)]
                    _, loss = sess.run([self.train_discriminator,self.d_loss], feed_dict={self.noise: X_noise, self.image: X_data})
                    print("discriminator loss: %f" % loss)
                    self.dis_loss.append(loss)

                X_noise = self.noise_generator(self.minibatch)
                _, loss, probs, img = sess.run([self.train_generator, self.g_loss, self.sample_probabilities, self.generated_img], feed_dict={self.noise: X_noise})
                print("generator loss: %f" % loss)
                self.gen_loss.append(loss)
                self.prob.append(np.mean(probs))
                if np.max(probs) >= 0.5:
                    self.save_fig(img[np.argmax(probs)], i)

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
            prob = sess.run(self.image_probability, feed_dict={self.noise: x})
            print("Probability of generated image is %f"%prob)
            plt.imshow(img[0].reshape((28,28)), cmap='Greys_r')

    def expected_probability(self, noise):
        '''

        :param noise:
        :return:
        '''
        gen_img = self.generator(noise)
        prob, _ = self.discriminator(gen_img)
        return tf.reduce_mean(prob)

    def discriminator_loss(self, noise, image):
        '''

        :param noise:
        :param image:
        :return:
        '''
        gen_image = self.generator(noise)
        gen_image_prob, _ = self.discriminator(gen_image)
        image_prob, _ = self.discriminator(image)
        return -tf.reduce_mean(tf.log(image_prob) + tf.log(1 - gen_image_prob))

    def generator_loss(self, noise):
        '''

        :param noise:
        :return:
        '''
        gen_image = self.generator(noise)
        gen_image_prob, _ = self.discriminator(gen_image)
        return -tf.reduce_mean(tf.log(gen_image_prob))

    def generator_nodes(self, x):
        '''

        :param x:
        :return:
        '''
        mu = 0
        sigma = 0.0001
        self.g_fc1_W = tf.Variable(xavier_init([100, 128]))#tf.truncated_normal(shape=(100, 128), mean=mu, stddev=sigma))
        self.g_fc1_b = tf.Variable(tf.zeros(128))

        self.g_fc2_W = tf.Variable(xavier_init([128,784]))#tf.truncated_normal(shape=(128, 784), mean=mu, stddev=sigma))
        self.g_fc2_b = tf.Variable(tf.zeros(784))

        self.theta_G = [self.g_fc1_W, self.g_fc1_W, self.g_fc2_W, self.g_fc2_b]


    def generator(self, x):
        '''

        :return:
        '''
        fc1 = tf.matmul(x, self.g_fc1_W) + self.g_fc1_b
        fc1 = tf.nn.relu(fc1)
        fc3 = tf.matmul(fc1, self.g_fc2_W) + self.g_fc2_b
        fc3 = tf.nn.tanh(fc3)

        return fc3


    def discriminator_nodes(self, x):
        '''

        :param x:
        :return:
        '''
        mu = 0
        sigma = 0.0001
        self.d_fc1_W = tf.Variable(xavier_init([784, 128]))#tf.truncated_normal(shape=(784, 128), mean=mu, stddev=sigma))
        self.d_fc1_b = tf.Variable(tf.zeros(128))

        self.d_fc2_W = tf.Variable(xavier_init([128,1]))#tf.truncated_normal(shape=(128, 1), mean=mu, stddev=sigma))
        self.d_fc2_b = tf.Variable(tf.zeros(1))

        self.theta_D = [self.d_fc1_W, self.d_fc1_b, self.d_fc2_W, self.d_fc2_b]

    def discriminator(self, x):
        '''

        :return:
        '''

        fc1 = tf.matmul(x, self.d_fc1_W) + self.d_fc1_b
        fc1 = tf.nn.relu(fc1)
        logit = tf.matmul(fc1, self.d_fc2_W) + self.d_fc2_b
        prob = tf.nn.sigmoid(logit)
        return prob, logit

    def noise_generator(self, m):
        '''

        :param m:
        :return:
        '''
        return np.random.uniform(-1, 1, size=(m,100))

    def save_fig(self, img, i):
        '''

        :param img:
        :return:
        '''
        if not os.path.exists('./gan-out/'):
            os.makedirs('./gan-out/')
        fig = plt.figure()
        plt.imshow(img.reshape(28, 28))
        plt.savefig('./gan-out/epoch-%d.png'%i)
        plt.close(fig)


    def get_cifar10(self):
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
            size = len(dict[b'data'])
            r = np.append(r, (dict[b'data'][:,:1024]).reshape((size, 32, 32)), axis=0)
            g = np.append(g, (dict[b'data'][:,1024:2048]).reshape((size, 32, 32)), axis=0)
            b = np.append(b, (dict[b'data'][:,2048:3072]).reshape((size, 32, 32)), axis=0)
        for i in range(r.shape[0]):
            img = np.dstack((r[i,:], g[i,:], b[i,:]))
            imgs.append(img)
        return np.array(imgs)/255.0

    def get_mnist(self):
        '''

        :return:
        '''
        mndata = MNIST('./datasets/')
        X_, _ = mndata.load_training()
        X_test, _ = mndata.load_testing()
        data = np.array(X_ + X_test)
        return data/255



g = Gan()
g.train()
g.test()
figure = plt.figure()
plt.subplot(3,1,1)
plt.plot(list(range(len(g.dis_loss))), g.dis_loss)
plt.ylabel("Discriminator loss")
plt.xlabel("Epoch")
plt.subplot(3,1,2)
plt.plot(list(range(len(g.gen_loss))), g.gen_loss)
plt.ylabel("Generator loss")
plt.xlabel("Epoch")
plt.subplot(3,1,3)
plt.plot(list(range(len(g.prob))), g.prob)
plt.ylabel("Expected probability")
plt.xlabel("Epoch")
plt.show()