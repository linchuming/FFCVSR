import tensorflow as tf
import numpy as np
import time

def inverse_warp(input, flow):
    shape = tf.shape(input)
    N = shape[0]
    H = shape[1]
    W = shape[2]
    # C = shape[3]
    N_i = tf.range(0, N)  # 0 .. N-1
    W_i = tf.range(0, W)
    H_i = tf.range(0, H)

    n, h, w = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
    n = tf.expand_dims(n, axis=3)  # [N, W, H, 1]
    h = tf.expand_dims(h, axis=3)
    w = tf.expand_dims(w, axis=3)

    n = tf.cast(n, tf.float32)
    h = tf.cast(h, tf.float32)
    w = tf.cast(w, tf.float32)

    v_col, v_row = tf.split(flow, 2, axis=-1)  # split flow into v_row & v_col
    """ calculate index """
    v_r0 = tf.floor(v_row)
    v_r1 = v_r0 + 1
    v_c0 = tf.floor(v_col)
    v_c1 = v_c0 + 1

    H_ = tf.cast(H - 1, tf.float32)
    W_ = tf.cast(W - 1, tf.float32)
    i_r0 = tf.clip_by_value(h + v_r0, 0., H_)
    i_r1 = tf.clip_by_value(h + v_r1, 0., H_)
    i_c0 = tf.clip_by_value(w + v_c0, 0., W_)
    i_c1 = tf.clip_by_value(w + v_c1, 0., W_)

    i_r0c0 = tf.cast(tf.concat([n, i_r0, i_c0], axis=-1), tf.int32)
    i_r0c1 = tf.cast(tf.concat([n, i_r0, i_c1], axis=-1), tf.int32)
    i_r1c0 = tf.cast(tf.concat([n, i_r1, i_c0], axis=-1), tf.int32)
    i_r1c1 = tf.cast(tf.concat([n, i_r1, i_c1], axis=-1), tf.int32)

    """ take value from index """
    f00 = tf.gather_nd(input, i_r0c0)
    f01 = tf.gather_nd(input, i_r0c1)
    f10 = tf.gather_nd(input, i_r1c0)
    f11 = tf.gather_nd(input, i_r1c1)

    """ calculate coeff """
    w00 = (v_r1 - v_row) * (v_c1 - v_col)
    w01 = (v_r1 - v_row) * (v_col - v_c0)
    w10 = (v_row - v_r0) * (v_c1 - v_col)
    w11 = (v_row - v_r0) * (v_col - v_c0)

    out = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11
    return out

def relu(inputs):
    return tf.nn.relu(inputs)

def leaky_relu(inputs):
    return tf.nn.leaky_relu(inputs, 0.2)

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
            for i in range(12):
                conv = self.res2d(conv, 'res' + str(i), 128)
            conv = self.conv2d(conv, 'conv1', 128)
            conv = conv + conv0

            feat = self.conv2d(conv, 'feat0', 128)
            feat = self.conv2d(feat, 'feat1', 128)
            feat = self.conv2d(feat, 'feat2', 128)

            conv = self.conv2d(conv, 'translation', 128)
            conv = self.deconv2d(conv, out_channels=1, ksize=8, stride=4, name='output')

            out = tf.add(conv, bic)
            return out, feat

    def refine_net(self, sr1, feat1, sr2, feat2, flow_s):
        with tf.variable_scope('refine_net', reuse=tf.AUTO_REUSE):
            flow = tf.image.resize_bilinear(flow_s, tf.shape(flow_s)[1:3] * 4) * 4.0
            sr1_to_sr2 = inverse_warp(sr1, flow)

            sr1_d = tf.space_to_depth(sr1_to_sr2, 4)
            sr2_d = tf.space_to_depth(sr2, 4)
            conv = tf.concat([sr1_d, sr2_d], axis=-1)

            conv = self.conv2d(conv, 'conv0', 128)
            conv = self.conv2d(conv, 'conv0_1', 128)
            conv0 = conv
            for i in range(4):
                conv = self.res2d(conv, 'res1_' + str(i), 128)
            conv = self.conv2d(conv, 'conv1', 128)
            conv = conv + conv0

            feat1 = inverse_warp(feat1, flow_s)

            conv = tf.concat([conv, feat1, feat2], axis=-1)
            conv = self.conv2d(conv, 'reduce', 128)
            conv0 = conv
            for i in range(4):
                conv = self.res2d(conv, 'res2_' + str(i), 128)
            conv = self.conv2d(conv, 'conv2', 128)
            conv = conv + conv0

            feat = self.conv2d(conv, 'feat0', 128)
            feat = self.conv2d(feat, 'feat1', 128)
            feat = self.conv2d(feat, 'feat2', 128)

            conv = self.conv2d(conv, 'translation', 128)
            conv = self.deconv2d(conv, out_channels=1, ksize=8, stride=4, name='output')

            out = tf.add(conv, sr2)
            return out, feat

    def flow_net(self, sr1, sr2):
        with tf.variable_scope('flow_net', reuse=tf.AUTO_REUSE):
            conv = tf.concat([sr1, sr2], axis=-1)

            conv = self.conv2d(conv, 'conv0_0', 32, act=leaky_relu)
            conv = self.conv2d(conv, 'conv0_1', 32, act=leaky_relu)
            s1 = tf.shape(conv)[1:3]
            conv = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            conv = self.conv2d(conv, 'conv1_0', 64, act=leaky_relu)
            conv = self.conv2d(conv, 'conv1_1', 64, act=leaky_relu)
            s2 = tf.shape(conv)[1:3]
            conv = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            conv = self.conv2d(conv, 'conv2_0', 128, act=leaky_relu)
            conv = self.conv2d(conv, 'conv2_1', 128, act=leaky_relu)
            s3 = tf.shape(conv)[1:3]
            conv = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            conv = self.conv2d(conv, 'conv3_0', 256, act=leaky_relu)
            conv = self.conv2d(conv, 'conv3_1', 256, act=leaky_relu)
            conv = tf.image.resize_bilinear(conv, s3)

            conv = self.conv2d(conv, 'conv4_0', 128, act=leaky_relu)
            conv = self.conv2d(conv, 'conv4_1', 128, act=leaky_relu)
            conv = tf.image.resize_bilinear(conv, s2)

            conv = self.conv2d(conv, 'conv5_0', 64, act=leaky_relu)
            conv = self.conv2d(conv, 'conv5_1', 64, act=leaky_relu)
            conv = tf.image.resize_bilinear(conv, s1)

            conv = self.conv2d(conv, 'conv6', 32, act=leaky_relu)
            conv = self.conv2d(conv, 'out', 2, act=tf.identity)
            return conv
