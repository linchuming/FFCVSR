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


import platform

if platform.uname()[0] != 'Windows':
    try:
        from custom_op.inverse_warp_op import inverse_warp
    except Exception:
        print('import custom_op.inverse_warp_op failed, and use default inverse_warp Op.')


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

    def extract_feature(self, clips_lr, t=5):
        with tf.variable_scope('extract_feature', reuse=tf.AUTO_REUSE):
            B, _, H, W, _ = tf.unstack(tf.shape(clips_lr))
            x = tf.reshape(clips_lr, [-1, H, W, 1])
            x = self.conv2d(x, 'conv1', 32)
            x = self.conv2d(x, 'conv2', 32, act=tf.identity)
            x = tf.reshape(x, [B, t, H, W, 32])
            return x

    def align_feature(self, clips_lr, motions, t=5):
        with tf.variable_scope('align_feature', reuse=tf.AUTO_REUSE):
            flows = []
            for i in range(t - 1):
                flows.append(motions[:, :, :, 2 * i: 2 * i + 2])
            p = 0
            res = []
            for i in range(t):
                x = clips_lr[:, i]
                if i != t // 2:
                    x = inverse_warp(x, flows[p])
                    p += 1
                res.append(x)
            return tf.concat(res, -1)

    def interp_frame(self, clips_lr, motions, t=5):
        interp_res = []
        flows = []
        for i in range(t-1):
            flows.append(motions[:, :, :, 2*i: 2*i+2])
        masks = tf.sigmoid(motions[:, :, :, 2 * (t-1):])
        for i in range((t-1) // 2):
            flow1 = flows[i]
            flow2 = flows[-(i+1)]
            mask1 = masks[:, :, :, i: i+1]
            mask2 = 1.0 - mask1
            im1 = clips_lr[:, i]
            im2 = clips_lr[:, -(i+1)]
            im1_warp = inverse_warp(im1, flow1)
            im2_warp = inverse_warp(im2, flow2)
            res = im1_warp * mask1 + im2_warp * mask2
            interp_res.append(res)
            interp_res.append(im1_warp)
            interp_res.append(im2_warp)
        return interp_res

    def flow_net(self, x, t=5):
        with tf.variable_scope('flow_net', reuse=tf.AUTO_REUSE):
            inp = []
            for i in range(t):
                inp.append(x[:, i])
            conv = tf.concat(inp, -1)

            conv = self.conv2d(conv, 'conv0_0', 64, act=leaky_relu)
            conv = self.conv2d(conv, 'conv0_1', 64, act=leaky_relu)
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

            conv = self.conv2d(conv, 'conv6', 64, act=leaky_relu)
            conv = self.conv2d(conv, 'out', 2 * (t-1) + (t-1) // 2, act=tf.identity)
            return conv

    def local_net(self, aligned_x, bic, t=5):
        with tf.variable_scope('local_net', reuse=tf.AUTO_REUSE):
            conv = tf.concat(aligned_x, axis=-1)
            conv = self.conv2d(conv, 'conv0', 128)
            conv0 = conv
            for i in range(8):
                conv = self.res2d(conv, 'res' + str(i), 128)
            conv = self.conv2d(conv, 'conv1', 128)
            conv = conv + conv0

            feat = self.conv2d(conv, 'feat0', 128)
            feat = self.conv2d(feat, 'feat1', 128, act=tf.tanh)
            # feat = self.conv2d(feat, 'feat2', 128)

            conv = self.conv2d(conv, 'translation', 128)
            conv = self.deconv2d(conv, out_channels=1, ksize=8, stride=4, name='output')

            out = tf.add(conv, bic)
            return out, feat

    def refine_net(self, sr1, feat1, sr2, feat2, motions, t=5):
        with tf.variable_scope('refine_net', reuse=tf.AUTO_REUSE):
            i = (t-1) // 2 - 1
            flow_s = motions[:, :, :, 2*i: 2*i+2]
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

            # feature gate
            att1 = self.conv2d(tf.concat([feat1, feat2], axis=-1), 'att1', 128)
            att2 = self.conv2d(att1, 'att2', 128, act=tf.sigmoid)
            att_feat = att2 * feat2 + (1.0 - att2) * feat1

            conv = tf.concat([conv, att_feat], axis=-1)
            conv = self.conv2d(conv, 'reduce', 128)
            conv0 = conv
            for i in range(4):
                conv = self.res2d(conv, 'res2_' + str(i), 128)
            conv = self.conv2d(conv, 'conv2', 128)
            conv = conv + conv0

            feat = self.conv2d(conv, 'feat0', 128)
            feat = self.conv2d(feat, 'feat1', 128, act=tf.tanh)
            # feat = self.conv2d(feat, 'feat2', 128)

            conv = self.conv2d(conv, 'translation', 128)
            conv = self.deconv2d(conv, out_channels=1, ksize=8, stride=4, name='output')

            out = tf.add(conv, sr2)
            return out, feat


if __name__ == '__main__':
    h = 480 // 4
    w = 720 // 4
    # h = 1080 // 4
    # w = 1920 // 4
    clips_lr = tf.ones([1, 5, h, w, 1])
    bic = tf.ones([1, h * 4, w * 4, 1])
    pre_feat = tf.ones([1, h, w, 128])
    pre_sr = tf.ones([1, h * 4, w * 4, 1])
    m = model()
    x = m.extract_feature(clips_lr)
    motions = m.flow_net(x)
    align_x = m.align_feature(x, motions)
    l_sr, l_feat = m.local_net(align_x, bic)
    sr, _ = m.refine_net(l_sr, l_feat, pre_sr, pre_feat, motions)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(sr)
    start = time.time()
    sess.run(sr)
    end = time.time()
    print(end - start)
