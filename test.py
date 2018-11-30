# -*- coding: UTF-8 -*-
import os
import struct
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 防止 tensorflow 的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
        print magic, n

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        print magic, num, rows, cols
    return images, labels


if __name__ == '__main__':
    images, labels = load_mnist('/home/ale/PycharmProjects/demo2/MNIST')
    fig, ax = plt.subplots( nrows=2, ncols=5, sharex=True, sharey=True, )

    ax = ax.flatten()
    for i in range(10):
        img = images[labels == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
