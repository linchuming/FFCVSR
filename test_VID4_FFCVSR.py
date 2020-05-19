import tensorflow as tf
from scipy import misc
import numpy as np
from utils import *
import os
from FFCVSR import model
import time

if __name__ == '__main__':
    save_img = False
    input_dir = 'datasets/VID4'
    addition_dir = 'original'
    output_dir = 'results/VID4/FFCVSR'
    datasets = ['calendar', 'city', 'walk', 'foliage']

    model_path = 'model_ckpt/ffcvsr.ckpt'
    update_T = 50

    sum_psnr = 0.0
    sum_time = 0.0
    sum_local_time = 0.0
    sum_local_psnr = 0.0
    sum_bic_psnr = 0.0
    for dataset in datasets:
        tf.reset_default_graph()
        input_path = os.path.join(input_dir, dataset, addition_dir)
        output_path = os.path.join(output_dir, dataset)
        scale = 4
        t = 5
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img_files = []
        for root, dirs, files in os.walk(input_path):
            img_files = sorted(files)
        hr_imgs = []
        lr_imgs = []
        bic_imgs = []
        height = width = 0
        for filename in img_files:
            img = misc.imread(os.path.join(input_path, filename))
            img = rgb2ycbcr(img)
            height, width, _ = img.shape
            height -= height % scale
            width -= width % scale
            img = img[:height, :width, :]
            hr_imgs.append(img)
            tmp = img[:, :, 0]
            lr_img = misc.imresize(tmp, [height // scale, width // scale], interp='bicubic', mode='F')
            lr_imgs.append(lr_img / 255.0)
            bic_img = misc.imresize(lr_img, [height, width], interp='bicubic', mode='F')
            bic_imgs.append(bic_img / 255.0)

        pad = t // 2
        lr_imgs = [lr_imgs[0]] * pad + lr_imgs + [lr_imgs[-1]] * pad
        bic_imgs = [bic_imgs[0]] * pad + bic_imgs + [bic_imgs[-1]] * pad
        print('files num:', len(lr_imgs))
        lr = tf.placeholder(dtype=tf.float32, shape=[1, t, height // scale, width // scale, 1])
        bic = tf.placeholder(dtype=tf.float32, shape=[1, height, width, 1])

        tf_pre_sr = tf.get_variable('tf_pre_sr',
                                    shape=[1, height, width, 1],
                                    dtype=tf.float32,
                                    collections=[tf.GraphKeys.LOCAL_VARIABLES])
        tf_pre_feat = tf.get_variable('tf_pre_feat',
                                      shape=[1, height // scale, width // scale, 128],
                                      dtype=tf.float32,
                                      collections=[tf.GraphKeys.LOCAL_VARIABLES])

        with tf.variable_scope('video_sr'):
            m = model()
            local_sr, local_feat = m.local_net(lr, bic)
            local_sr = tf.clip_by_value(local_sr, 0, 1)
            refine_sr, refine_feat = m.refine_net(tf_pre_sr, tf_pre_feat, local_sr, local_feat)
            refine_sr = tf.clip_by_value(refine_sr, 0, 1)

        saver = tf.train.Saver()

        with tf.control_dependencies([local_sr, local_feat]):
            assign_local_to_pre = tf.group(
                tf.assign(tf_pre_sr, local_sr),
                tf.assign(tf_pre_feat, local_feat)
            )

        with tf.control_dependencies([refine_sr, refine_feat]):
            assign_refine_to_pre = tf.group(
                tf.assign(tf_pre_sr, refine_sr),
                tf.assign(tf_pre_feat, refine_feat)
            )

        # # model analysis
        # tf.contrib.tfprof.model_analyzer.print_model_analysis(
        #     tf.get_default_graph(),
        #     tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

        with tf.Session() as sess:
            saver.restore(sess, model_path)
            avg_psnr = []
            avg_time = []
            avg_local_time = []
            avg_local_psnr = []
            avg_bic_psnr = []
            num = 0
            for i in range(1, len(lr_imgs) - t + 2):
                lrs = []
                bics = []
                for j in range(i - 1, i + t - 1):
                    lrs.append(lr_imgs[j])
                    bics.append(bic_imgs[j])
                lrs = np.stack(lrs).astype(np.float32)
                lrs = np.expand_dims(lrs, axis=0)
                lrs = np.expand_dims(lrs, axis=4)
                bics = np.stack(bics).astype(np.float32)
                bics = np.expand_dims(bics, axis=0)
                bics = np.expand_dims(bics, axis=4)

                concat_lr = np.concatenate([lrs])
                concat_bic = np.concatenate([bics[:, t // 2, :, :, :]])
                if i == 1:
                    out = sess.run([local_sr, assign_local_to_pre], feed_dict={lr: concat_lr, bic: concat_bic})
                    out = sess.run([local_sr, refine_sr, assign_refine_to_pre], feed_dict={lr: concat_lr, bic: concat_bic})
                start = time.time()
                if i == 1:
                    out, l_feat = sess.run([local_sr, assign_local_to_pre], feed_dict={lr: concat_lr, bic: concat_bic})
                    print('local time:', time.time() - start)
                    local_out = out
                elif (i - 1) % update_T == 0:
                    l_sr, out, _ = sess.run([local_sr, refine_sr, assign_local_to_pre],
                                                         feed_dict={
                                                             lr: concat_lr,
                                                             bic: concat_bic
                                                         })
                    local_out = l_sr
                else:
                    l_sr, out, _ = sess.run([local_sr, refine_sr, assign_refine_to_pre],
                                                         feed_dict={
                                                             lr: concat_lr,
                                                             bic: concat_bic
                                                         })
                    local_out = l_sr

                end = time.time()
                avg_time.append(end - start)
                out1 = out[0, :, :, 0]
                out1 = np.clip(out1, 0, 1)

                out2 = local_out[0, :, :, 0]
                out2 = np.clip(out2, 0, 1)

                height, width = out1.shape
                img = out1 * 255.0
                img = np.clip(img, 16, 235)

                local_img = out2 * 255.0
                local_img = np.clip(local_img, 16, 235)

                output_name = '%04d.png' % (i)
                hr_img = hr_imgs[i - 1]
                psnr_val = psnr(img[scale:height - scale, scale:width - scale],
                                hr_img[scale:height - scale, scale:width - scale, 0])[0]

                local_psnr_val = psnr(local_img[scale:height - scale, scale:width - scale],
                                      hr_img[scale:height - scale, scale:width - scale, 0])[0]

                bic_y = concat_bic[0, :, :, 0] * 255.0
                bic_psnr_val = psnr(bic_y[scale:-scale, scale:-scale],
                                    hr_img[scale:-scale, scale:-scale, 0])[0]

                print(output_name, psnr_val, local_psnr_val, bic_psnr_val)
                avg_psnr.append(psnr_val)
                avg_local_psnr.append(local_psnr_val)
                avg_bic_psnr.append(bic_psnr_val)
                num += 1

                lr_img = ycbcr2rgb(hr_img)
                lr_img = img_to_uint8(lr_img)
                lr_img = misc.imresize(lr_img, [height // scale, width // scale], interp='bicubic')
                bic_img = misc.imresize(lr_img, [height, width], interp='bicubic')
                bic_img = np.float64(bic_img)
                bic_img = rgb2ycbcr(bic_img)
                bic_img[:, :, 0] = img
                rgb_img = ycbcr2rgb(bic_img)

                if save_img:
                    misc.imsave(os.path.join(output_path, output_name), img_to_uint8(rgb_img))
        print(dataset)
        avg_psnr = np.mean(avg_psnr[2:-2])
        avg_local_psnr = np.mean(avg_local_psnr[2:-2])
        avg_time = np.mean(avg_time[2:-2])
        avg_bic_psnr = np.mean(avg_bic_psnr[2:-2])
        print('avg psnr:', avg_psnr)
        print('avg local psnr:', avg_local_psnr)
        print('avg bicubic psnr:', avg_bic_psnr)
        print('avg time:', avg_time)
        sum_psnr += avg_psnr
        sum_time += avg_time
        sum_local_psnr += avg_local_psnr
        sum_bic_psnr += avg_bic_psnr


    print('Summary:')
    print('avg psnr:', sum_psnr / len(datasets))
    print('avg local psnr:', sum_local_psnr / len(datasets))
    print('avg bicubic psnr:', sum_bic_psnr / len(datasets))
    print('avg time:', sum_time / len(datasets))

