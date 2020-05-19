import tensorflow as tf
import logging
from FFCVSR_motion import model
import datetime
import os
from scipy import misc

checkpoint_dir = 'output_reds/FFCVSR_motion/model'
log_dir = 'output_reds/FFCVSR_motion/log'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

record_path = 'tfrecords/REDS'
record_num = 10
length = 15
H = 180
W = 180
patch_H = patch_W = 180
scale = 4
batch_size = 4
train_num = 106200
step_one_epoch = train_num // batch_size
max_step_num = 350000
t = 5
local_net_times = length - t + 1
lr_bounds = [300000]
lr_values = [1e-4, 1e-5]


def _parse_gt(gt, scale=4):
    gt = gt[:, :, 0]
    lr = misc.imresize(gt, 1.0 / scale, interp='bicubic')
    bic = misc.imresize(lr, scale * 1.0, interp='bicubic')
    return lr, bic


def _parse_one_example(example):
    features = tf.parse_single_example(
        example,
        features={
            'gt': tf.FixedLenFeature([], tf.string)
        }
    )

    gt = features['gt']

    gt = tf.decode_raw(gt, tf.uint8)
    gt = tf.reshape(gt, [length, H, W, 1])

    if patch_H < H or patch_W < W:
        # random crop
        rnd_h = tf.random_uniform([], 0, H - patch_H + 1, tf.int32)
        rnd_w = tf.random_uniform([], 0, W - patch_W + 1, tf.int32)
        gt = gt[:, rnd_h: rnd_h + patch_H, rnd_w: rnd_w + patch_W]

    clip_lr = []
    clip_bic = []
    for i in range(length):
        lr, bic = tf.py_func(lambda x: _parse_gt(x, scale), [gt[i]], [tf.uint8, tf.uint8])
        clip_lr.append(lr)
        clip_bic.append(bic)
    clip_lr = tf.stack(clip_lr)
    clip_bic = tf.stack(clip_bic)

    clip_lr = tf.reshape(clip_lr, [length, patch_H // scale, patch_W // scale, 1])
    clip_bic = tf.reshape(clip_bic, [length, patch_H, patch_W, 1])

    gt = tf.cast(gt, tf.float32) / 255.0
    clip_lr = tf.cast(clip_lr, tf.float32) / 255.0
    clip_bic = tf.cast(clip_bic, tf.float32) / 255.0

    clip_lr, clip_bic, gt = tf.cond(tf.random_uniform([], 0, 1) < 0.5,
                                    lambda: (clip_lr[:, ::-1, :], clip_bic[:, ::-1, :], gt[:, ::-1, :]),
                                    lambda: (clip_lr, clip_bic, gt))
    clip_lr, clip_bic, gt = tf.cond(tf.random_uniform([], 0, 1) < 0.5,
                                    lambda: (clip_lr[:, :, ::-1, :], clip_bic[:, :, ::-1, :], gt[:, :, ::-1, :]),
                                    lambda: (clip_lr, clip_bic, gt))
    clip_lr, clip_bic, gt = tf.cond(tf.random_uniform([], 0, 1) < 0.5,
                                    lambda: (clip_lr[::-1, :], clip_bic[::-1, :], gt[::-1, :]),
                                    lambda: (clip_lr, clip_bic, gt))
    clip_lr, clip_bic, gt = tf.cond(tf.random_uniform([], 0, 1) < 0.5,
                                    lambda: (tf.image.rot90(clip_lr), tf.image.rot90(clip_bic), tf.image.rot90(gt)),
                                    lambda: (clip_lr, clip_bic, gt))

    return clip_lr, clip_bic, gt


def read_train_data(batch_size, shuffle_num):
    filenames = []
    for i in range(record_num):
        filenames.append(os.path.join(record_path, 'data%i.tfrecords') % i)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_one_example, num_parallel_calls=8)
    dataset = dataset.shuffle(shuffle_num).prefetch(shuffle_num // 4)
    dataset = dataset.batch(batch_size).repeat()

    iterator = dataset.make_one_shot_iterator()
    clip_lr, clip_bic, gt = iterator.get_next()
    return clip_lr, clip_bic, gt


def restore_session_from_checkpoint(sess, saver):
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint:
        logging.info("Restore session from checkpoint: {}".format(checkpoint))
        saver.restore(sess, checkpoint)
        return True
    else:
        return False


def main(unused_argv):
    logging.basicConfig(level=logging.INFO)
    tf.reset_default_graph()

    global_step = tf.Variable(0, name="global_step", trainable=False)

    clip_lr, clip_bic, gt = read_train_data(batch_size, 3000)

    # train model build #
    with tf.variable_scope('video_sr'):
        m = model()
        l_sr = []
        f_sr = []
        l_feat = []
        l_motions = []
        l_interp = []
        # setup local net
        for i in range(0, local_net_times):
            clips_x = m.extract_feature(clip_lr[:, i:i + t])
            motions = m.flow_net(clips_x)
            aligned_x = m.align_feature(clips_x, motions)
            out, feat = m.local_net(aligned_x, clip_bic[:, i + t // 2])
            l_sr.append(out)
            l_feat.append(feat)
            l_motions.append(motions)
            interp_res = m.interp_frame(clip_lr[:, i:i + t], motions)
            l_interp.append(interp_res)

        # setup context net
        pre_sr = l_sr[0]
        pre_feat = l_feat[0]
        for i in range(1, local_net_times):
            sr1 = tf.clip_by_value(pre_sr, 0, 1)
            sr2 = tf.clip_by_value(l_sr[i], 0, 1)
            out, feat = m.refine_net(sr1, pre_feat, sr2, l_feat[i], l_motions[i])
            f_sr.append(out)
            pre_sr = out
            pre_feat = feat

    with tf.name_scope('train'):
        # calculate l2_loss for local net
        g_loss = []
        for i in range(0, local_net_times):
            g_loss.append(tf.reduce_mean(tf.nn.l2_loss(l_sr[i] - gt[:, i + t // 2])))
        g_loss = tf.add_n(g_loss) / local_net_times / batch_size * 2

        # calculate l2_loss for context net
        f_loss = []
        for i in range(1, local_net_times):
            f_loss.append(tf.reduce_mean(tf.nn.l2_loss(f_sr[i - 1] - gt[:, i + t // 2])))

        f_loss = tf.add_n(f_loss) / (local_net_times - 1) / batch_size * 2

        # calculate interp loss for flow net
        interp_loss = []
        i_count = 0
        for i in range(0, local_net_times):
            interp_res = l_interp[i]
            for im in interp_res:
                interp_loss.append(tf.reduce_mean(tf.nn.l2_loss(im - clip_lr[:, i + t // 2])))
                i_count += 1.0
        interp_loss = tf.add_n(interp_loss) / i_count / batch_size

        # calculate l2 regulation loss for all weights
        weights_norm = tf.reduce_sum(
            1e-5 * tf.stack(
                [tf.nn.l2_loss(i) for i in tf.get_collection('weights')]
            )
        )
        loss = g_loss + f_loss + interp_loss + weights_norm

        # setup learning rate
        learning_rate = tf.train.piecewise_constant(global_step, lr_bounds, lr_values)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.name_scope('summaries'):
        tf.summary.scalar('global_step', global_step)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('f_loss', f_loss)
        tf.summary.scalar('interp_loss', interp_loss)
        tf.summary.scalar('weight_reg', weights_norm)
        tf.summary.scalar('loss', loss)

        for i in range(1, local_net_times):
            tf.summary.image('%dbic' % i, clip_bic[0, i + t // 2:], 1)
            tf.summary.image('%dl_f' % i, tf.clip_by_value(l_sr[i], 0, 1), 1)
            tf.summary.image('%df_f' % i, tf.clip_by_value(f_sr[i - 1], 0, 1), 1)
            tf.summary.image('%dg_f' % i, gt[0, i + t // 2:], 1)

        summary_op = tf.summary.merge_all()

    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    saver = tf.train.Saver(max_to_keep=500)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init_op)

    restore_session_from_checkpoint(sess, saver)
    start_time = datetime.datetime.now()
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    avg_loss = 0.0
    local_step = 0
    while True:
        _, loss_value, step = sess.run([train_op, loss, global_step])
        avg_loss += loss_value
        local_step += 1
        if step % 5000 == 0:
            saver.save(sess, os.path.join(checkpoint_dir, 'checkpoint.ckpt'), global_step=step)
        if step % step_one_epoch == 0:
            # saver.save(sess, os.path.join(FLAGS.train_dir, 'checkpoint.ckpt'), global_step=step)
            avg_loss = loss_value
            local_step = 1
        if step % 20 == 0:
            end_time = datetime.datetime.now()
            summary_value = sess.run(summary_op)
            logging.info("[{}] Step:{}, loss:{}, avg_loss:{}".format(
                end_time - start_time, step, loss_value, avg_loss / local_step
            ))
            writer.add_summary(summary_value, step)
            start_time = end_time

        if step >= max_step_num:
            logging.info("Done train.")
            break


if __name__ == '__main__':
    tf.app.run()
