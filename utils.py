import os
import pickle

import numpy as np

from mnist import MNIST


def get_mnist():
    '''

    :return:
    '''
    mndata = MNIST('./datasets/')
    X_, _ = mndata.load_training()
    X_test, _ = mndata.load_testing()
    data = np.array(X_ + X_test)
    return data / 255.0

def get_cifar10():
    '''

    :return:
    '''
    source = '../data/cifar-10-batches-py/'
    files = ["%s//%s" % (source, f) for f in os.listdir(source) if not (f.endswith('.html') or f.endswith('.meta'))]
    r = np.empty((0, 32, 32), dtype=np.uint8)
    g = np.empty((0, 32, 32), dtype=np.uint8)
    b = np.empty((0, 32, 32), dtype=np.uint8)
    imgs = []
    for file in files:
        dict = pickle.load(open(file, 'rb'), encoding='bytes')
        size = len(dict[b'data'])
        r = np.append(r, (dict[b'data'][:, :1024]).reshape((size, 32, 32)), axis=0)
        g = np.append(g, (dict[b'data'][:, 1024:2048]).reshape((size, 32, 32)), axis=0)
        b = np.append(b, (dict[b'data'][:, 2048:3072]).reshape((size, 32, 32)), axis=0)
    for i in range(r.shape[0]):
        img = np.dstack((r[i, :], g[i, :], b[i, :]))
        imgs.append(img)
    return np.array(imgs) / 255.0
