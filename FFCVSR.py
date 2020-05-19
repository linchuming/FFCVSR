import tensorflow as tf
import numpy as np
import time


def relu(inputs):
    return tf.nn.relu(inputs)


class model():
    def conv2d(self, inputs, name, out_channels, act=relu, ksize=3):
        with tf.variable_scope(name):
            in_channels = inputs.get_shape()[-1]
            filter = tf.get_variable('weight', shape=[ksize, ksize, in_channels, out_channels],
                                     initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(inputs, filter, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.get_variable('bias', shape=[out_channels], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)
            conv = act(conv)

            tf.add_to_collection('weights', filter)
            return conv

    def deconv2d(self, inputs, name, out_channels, ksize, stride):
        with tf.variable_scope(name):
            input_shape = inputs.get_shape()
            in_channels = input_shape[-1]
            input_shape = tf.shape(inputs)
            filter = tf.get_variable('weight', shape=[ksize, ksize, out_channels, in_channels],
                                     initializer=tf.contrib.layers.xavier_initializer())
            output_shape = [input_shape[0], input_shape[1] * stride, input_shape[2] * stride, out_channels]
            deconv = tf.nn.conv2d_transpose(inputs, filter, output_shape, [1, stride, stride, 1])
            bias = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, bias)

            tf.add_to_collection('weights', filter)
            return deconv

    def res2d(self, inputs, name, out_channels, ksize=3, scale=0.1, act=tf.identity):
        with tf.variable_scope(name):
            conv = self.conv2d(inputs, 'conv1', out_channels, ksize=ksize)
            conv = self.conv2d(conv, 'conv2', out_channels, ksize=ksize, act=tf.identity)
            return act(inputs + conv * scale)

    def local_net(self, clips_lr, bic):
        with tf.variable_scope('local_net', reuse=tf.AUTO_REUSE):
            inputs = []
            for i in range(5):
                inputs.append(clips_lr[:, i])
            conv = tf.concat(inputs, axis=-1)
            conv = self.conv2d(conv, 'conv0', 128)
            conv0 = conv
            for i in range(8):
                conv = self.res2d(conv, 'res' + str(i), 128)
            conv = self.conv2d(conv, 'conv1', 128)
            conv = conv + conv0

            feat = self.conv2d(conv, 'feat0', 128)
            feat = self.conv2d(feat, 'feat1', 128)

            conv = self.conv2d(conv, 'translation', 128)
            conv = tf.concat([conv0, feat, conv], -1)
            conv = self.deconv2d(conv, out_channels=1, ksize=8, stride=4, name='output')

            out = tf.add(conv, bic)
            return out, feat

    def refine_net(self, sr1, feat1, sr2, feat2):
        with tf.variable_scope('refine_net', reuse=tf.AUTO_REUSE):
            sr1_d = tf.space_to_depth(sr1, 4)
            sr2_d = tf.space_to_depth(sr2, 4)
            conv = tf.concat([sr1_d, sr2_d], axis=-1)

            conv = self.conv2d(conv, 'conv0', 128)
            conv0 = conv
            for i in range(2):
                conv = self.res2d(conv, 'res1_' + str(i), 128)
            conv = self.conv2d(conv, 'conv1', 128)
            conv = conv + conv0

            conv = tf.concat([conv, feat1, feat2], axis=-1)
            conv = self.conv2d(conv, 'reduce', 128)
            conv0 = conv
            for i in range(2):
                conv = self.res2d(conv, 'res2_' + str(i), 128)
            conv = self.conv2d(conv, 'conv2', 128)
            conv = conv + conv0

            feat = self.conv2d(conv, 'feat0', 128)
            feat = self.conv2d(feat, 'feat1', 128)

            conv = self.conv2d(conv, 'translation', 128)
            conv = tf.concat([conv0, feat, conv], -1)
            conv = self.deconv2d(conv, out_channels=1, ksize=8, stride=4, name='output')

            out = tf.add(conv, sr2)
            return out, feat


if __name__ == '__main__':
    clips_lr = tf.placeholder(tf.float32, [None, 5, 144, 180, 1])
    bic = tf.placeholder(tf.float32, [None, 576, 720, 1])
    pre_feat = tf.placeholder(tf.float32, [None, 144, 180, 128])
    pre_sr = tf.placeholder(tf.float32, [None, 576, 720, 1])
    m = model()
    l_sr, l_feat = m.local_net(clips_lr, bic)
    sr, _ = m.refine_net(l_sr, l_feat, pre_sr, pre_feat)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(sr, feed_dict={clips_lr: np.zeros([1, 5, 144, 180, 1]), bic: np.zeros([1, 576, 720, 1]),
                            pre_feat: np.zeros([1, 144, 180, 128]), pre_sr: np.zeros([1, 576, 720, 1])})
    start = time.time()
    sess.run(sr, feed_dict={clips_lr: np.zeros([1, 5, 144, 180, 1]), bic: np.zeros([1, 576, 720, 1]),
                            pre_feat: np.zeros([1, 144, 180, 128]), pre_sr: np.zeros([1, 576, 720, 1])})
    end = time.time()
    print(end - start)
