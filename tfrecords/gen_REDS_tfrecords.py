import tensorflow as tf
import os
import numpy as np
from scipy import misc
from skimage.color import rgb2ycbcr
import random
import glob
import tqdm

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

num = 0
data_path = 'datasets/REDS/'
if __name__ == '__main__':
    save_path = 'REDS/data%d.tfrecords'
    recorder_num = 10
    clip_len = 15
    patch_h = 180
    patch_w = 180
    crop_step = 128
    writers = []
    for i in range(recorder_num):
       writers.append(tf.python_io.TFRecordWriter(save_path % i))

    dirs = os.listdir(data_path)
    for d in tqdm.tqdm(dirs[:]):
        pngs = sorted(glob.glob(os.path.join(data_path, d, '*.png')))
        for i in range(0, len(pngs) - clip_len + 1, clip_len):
            png_paths = pngs[i: i+clip_len]

            imgs = []
            for png_path in png_paths:
                img = misc.imread(png_path)
                img = rgb2ycbcr(img)
                imgs.append(img[:, :, 0])

            def gen_patchs_with_scale(scale):
                assert len(imgs) == clip_len
                ims = []
                height, width = imgs[0].shape
                height = np.int32(np.round(height / scale))
                width = np.int32(np.round(width / scale))
                # print(height, width, len(imgs))
                for i in range(len(imgs)):
                    ims.append(misc.imresize(imgs[i], [height, width], interp='bicubic'))
                for row in range(0, height - patch_h + 1, crop_step):
                    for col in range(0, width - patch_w + 1, crop_step):
                        gts = []
                        inputs = []
                        lrs = []
                        for j in range(0, clip_len):
                            patch = ims[j][row:row + patch_h, col:col + patch_w]
                            assert patch.dtype == np.uint8

                            # print(patch.dtype)
                            gts.append(patch)

                        tf_gt = np.stack(gts, axis=0)
                        tf_gt = tf_gt.astype(np.uint8)
                        # print(np.sum(np.square(tf_input - tf_gt)))
                        # exit()
                        assert tf_gt.dtype == np.uint8
                        assert tf_gt.shape[:] == (clip_len, patch_h, patch_w)
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'gt': _bytes_feature(tf_gt.tostring())
                        }))
                        global num
                        writers[num % recorder_num].write(example.SerializeToString())
                        num += 1

            gen_patchs_with_scale(1)
            gen_patchs_with_scale(1.5)
            gen_patchs_with_scale(2)
            gen_patchs_with_scale(3)
            gen_patchs_with_scale(4)

    print('example num: %d' % num)
    for i in range(recorder_num):
        writers[i].close()


