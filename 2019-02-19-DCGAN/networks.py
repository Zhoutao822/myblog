from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

ds = tf.contrib.distributions
layers = tf.contrib.layers
tfgan = tf.contrib.gan

num_classes = 10
batch_size = 64
learning_rate = 2e-4
num_threads = 5
s_h4 = 7
s_w4 = 7
s_h2 = 14
s_w2 = 14
s_h = 28
s_w = 28

gf_dim = 64
d_h = 2
d_w = 2
c_dim = 1
df_dim = 64
dfc_dim=1024

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train , x_test = x_train / 255. , x_test / 255.
    x_train = tf.reshape(x_train, [-1, 28, 28, 1])
    x_test = tf.reshape(x_test, [-1, 28, 28, 1])
    y_train = tf.one_hot(y_train, depth=10, axis=-1)
    y_test = tf.one_hot(y_test, depth=10, axis=-1)

    images_train, labels_train = tf.train.batch(
        [x_train, y_train],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=5 * batch_size)

    images_test, labels_test = tf.train.batch(
        [x_test, y_test],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=5 * batch_size)

    return images_train, labels_train, images_test, labels_test

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return tf.concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

_leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.2)

def mnist_generator(inputs, is_training=True):
    z, y = inputs
    with tf.contrib.framework.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=tf.nn.relu, 
        normalizer_fn=layers.batch_norm,
        strides=[1, d_h, d_w, 1]):
        with tf.contrib.framework.arg_scope(
            [layers.batch_norm], 
            is_training=is_training, 
            decay=0.9,
            updates_collections=None,
            epsilon=1e-5,
            scale=True):

            yb = tf.reshape(y, [batch_size, 1, 1, num_classes])
            z = tf.concat([z, y], 1)
            print("yb shape: {}".format(yb.shape))
            # yb shape: (64, 1, 1, 10)
            print("Z shape: {}".format(z.shape))
            # Z shape: (64, 110)
            h0 = layers.fully_connected(z, 1024)
            h0 = tf.concat([h0, y], 1)
            print("y shape: {}".format(y.shape))
            # y shape: (64, 10)
            print("h0 shape: {}".format(h0.shape))
            # h0 shape: (64, 1034)
            h1 = layers.fully_connected(h0, s_h4 * s_w4 * 2* gf_dim)
            h1 = tf.reshape(h1, [batch_size, s_h4, s_w4, gf_dim * 2])

            h1 = conv_cond_concat(h1, yb)

            h2 = layers.conv2d_transpose(h1, gf_dim * 2, [batch_size, s_h2, s_w2, gf_dim * 2])

            h2 = conv_cond_concat(h2, yb)

            h3 = layers.conv2d_transpose(h2, c_dim, [batch_size, s_h, s_w, c_dim], normalizer_fn=None, activation_fn=tf.sigmoid)

            return h3

def mnist_discriminator(img, y):
    with tf.contrib.framework.arg_scope(
        [layers.conv2d, layers.fully_connected],
        activation_fn=_leaky_relu, 
        normalizer_fn=layers.batch_norm,
        strides=[1, d_h, d_w, 1]):

        yb = tf.reshape(y, [batch_size, 1, 1, num_classes])
        x = conv_cond_concat(img, yb)

        h0 = layers.conv2d(x, c_dim + num_classes, [5, 5], normalizer_fn=None)
        h0 = conv_cond_concat(h0, yb)

        h1 = layers.conv2d(h0, df_dim + num_classes, [5, 5])
        h1 = tf.reshape(h1, [batch_size, -1])      
        h1 = tf.concat([h1, y], 1)
        
        h2 = layers.fully_connected(h1, dfc_dim)
        h2 = tf.concat([h2, y], 1)

        h3 = layers.fully_connected(h2, 1, activation_fn=tf.sigmoid)
        
        return h3
