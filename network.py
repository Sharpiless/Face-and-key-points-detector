import tensorflow as tf

from read_data import Reader

import os

import numpy as np


slim = tf.contrib.slim


class Net(object):

    def __init__(self, is_training=True):

        self.is_training = is_training

        if self.is_training:

            self.reader = Reader()

        self.batch_size = 16

        self.lr = 2e-4

        self.wd = 5e-3

        self.epoches = 100

        self.batches = 64

        self.size = 96

        self.label_num = 30

        self.x = tf.placeholder(tf.float32, [None, self.size, self.size, 1])

        self.y = tf.placeholder(tf.float32, [None, self.label_num])

        self.y_hat = self.network(self.x)

        self.model_path = './model'

        self.ckpt_path = os.path.join(self.model_path, 'model.ckpt')

        self.saver = tf.train.Saver()

    def loss_layer(self, y, y_hat):

        loss = tf.reduce_sum(tf.square(y - y_hat))

        return loss

    def network(self, inputs):

        with tf.variable_scope('net'):

            with slim.arg_scope([slim.conv2d],

                                activation_fn=tf.nn.relu,

                                weights_regularizer=slim.l2_regularizer(self.wd)):

                # Block init

                net = slim.conv2d(inputs, 1024, [3, 3],

                                  2, scope='conv_init', padding='SAME')

                # Block 1

                net = slim.repeat(net, 2, slim.conv2d,

                                  64, [3, 3], scope='conv1', padding='SAME')

                net = slim.max_pool2d(

                    net, [2, 2], scope='pool1', padding='SAME')

                net = tf.layers.batch_normalization(

                    net, trainable=self.is_training, name='BN_block1')

                # Block 2

                net = slim.repeat(net, 2, slim.conv2d,

                                  128, [3, 3], scope='conv2')

                net = slim.max_pool2d(

                    net, [2, 2], scope='pool2', padding='SAME')

                net = tf.layers.batch_normalization(

                    net, trainable=self.is_training, name='BN_block2')

                # Block 3

                net = slim.repeat(net, 3, slim.conv2d,

                                  256, [3, 3], scope='conv3')

                net = slim.max_pool2d(

                    net, [2, 2], scope='pool3', padding='SAME')

                net = tf.layers.batch_normalization(

                    net, trainable=self.is_training, name='BN_block3')

                # Block 4

                net = slim.repeat(net, 3, slim.conv2d,

                                  512, [3, 3], scope='conv4')

                net = slim.max_pool2d(

                    net, [2, 2], scope='pool4', padding='SAME')

                net = tf.layers.batch_normalization(

                    net, trainable=self.is_training, name='BN_block4')

                # Block 5

                net = slim.repeat(net, 3, slim.conv2d,

                                  512, [3, 3], scope='conv5')

                net = tf.layers.batch_normalization(

                    net, trainable=self.is_training, name='BN_block5')

                # Block 6

                net = slim.conv2d(net, 1024, [3, 3],

                                  2, scope='conv6')

                net = tf.layers.batch_normalization(

                    net, trainable=self.is_training, name='BN_block6')

                net = tf.layers.flatten(net)

                logits = tf.layers.dense(net, self.label_num)

                if self.is_training:

                    logits = tf.layers.dropout(logits, rate=1-0.2)

                # logits = tf.nn.tanh(logits)

                return logits

    def train_net(self):

        if not os.path.exists(self.model_path):

            os.makedirs(self.model_path)

        self.loss = self.loss_layer(self.y, self.y_hat)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)

        self.train_step = self.optimizer.minimize(self.loss)

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(self.model_path)

            if ckpt and ckpt.model_checkpoint_path:

                # 如果保存过模型，则在保存的模型的基础上继续训练

                self.saver.restore(sess, ckpt.model_checkpoint_path)

                print('Model Reload Successfully!')

            for epoch in range(self.epoches):

                loss_list = []

                for batch in range(self.batch_size):

                    images, labels = self.reader.generate(self.batch_size)

                    feed_dict = {

                        self.x: images,

                        self.y: labels

                    }

                    loss_value = sess.run(self.loss, feed_dict)
                    _ = sess.run(self.train_step, feed_dict)

                    loss_list.append(loss_value)

                loss = np.mean(np.array(loss_list))

                self.saver.save(sess, self.ckpt_path)

                print('epoch:{} loss:{}'.format(epoch, loss))

                with open('./losses.txt', 'a') as f:

                    f.write(str(loss)+'\n')

    def test_net(self, image, sess):

        image = image.reshape((1, self.size, self.size, 1)) - 127.5

        points = sess.run(self.y_hat, feed_dict={self.x: image})

        points = (points * self.size).astype(np.int)

        return np.squeeze(points)


if __name__ == '__main__':

    import cv2

    import matplotlib.pyplot as plt

    net = Net()

    net.train_net()

    with open('./losses.txt', 'r') as f:

        losses = f.read().splitlines()

    losses = [eval(v) for v in losses]

    plt.plot(losses)

    plt.title('loss')

    plt.show()
