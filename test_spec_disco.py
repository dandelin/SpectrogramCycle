from __future__ import absolute_import, division, print_function

import os
import batching_spectrograms_single
import utils
import models_spec_disco
import argparse
import numpy as np
import tensorflow as tf
import image_utils as im

from glob import glob

def minmax_normalize(tensor):
    min = np.min(tensor)
    max = np.max(tensor)
    a = tensor - min
    b = max - min
    return (a / b) * 2 - 1

""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', dest='dataset', default='horse2zebra', help='which dataset to use')
args = parser.parse_args()

dataset = args.dataset
a_test_path, b_test_path = ['./datasets/' + dataset + '/test{}/'.format(e) for e in ['A', 'B']]

height, width = 512, 128
batch_size = 1

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

""" run """
with tf.Session(config=config) as sess:
    a_real = tf.placeholder(tf.float32, shape=[None, height, width, 1])
    b_real = tf.placeholder(tf.float32, shape=[None, height, width, 1])

    a2b = models_spec_disco.generator(a_real, 'a2b')
    b2a = models_spec_disco.generator(b_real, 'b2a')
    b2a2b = models_spec_disco.generator(b2a, 'a2b', reuse=True)
    a2b2a = models_spec_disco.generator(a2b, 'b2a', reuse=True)

    # restore
    saver = tf.train.Saver()
    ckpt_path = utils.load_checkpoint('./checkpoints/' + dataset + '_disco', sess, saver)
    if ckpt_path is None:
        raise Exception('No checkpoint!')
    else:
        print('Copy variables from % s' % ckpt_path)

    # test

    save_dir = './test_predictions/' + dataset + '_disco/'
    utils.mkdir([save_dir])

    a_test_pool = batching_spectrograms_single.SpecData(sess, a_test_path, batch_size, retain_all=True)
    b_test_pool = batching_spectrograms_single.SpecData(sess, b_test_path, batch_size, retain_all=True)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    a_final = []
    ab_final = []
    aba_final = []
    b_final = []
    ba_final = []
    bab_final = []
    for i in range(50):
        print('learning a, ', i)
        a_real_ipt = a_test_pool.batch()
        a2b_opt, a2b2a_opt = sess.run([a2b, a2b2a], feed_dict={a_real: a_real_ipt})
        ab_final.append(a2b_opt)
        aba_final.append(a2b2a_opt)
        a_final.append(a_real_ipt)

    for i in range(50):
        print('learning b, ', i)
        b_real_ipt = b_test_pool.batch()
        b2a_opt, b2a2b_opt = sess.run([b2a, b2a2b], feed_dict={b_real: b_real_ipt})
        ba_final.append(b2a_opt)
        bab_final.append(b2a2b_opt)
        b_final.append(b_real_ipt)

    for i, (a, ab, aba, b, ba, bab) in enumerate(zip(a_final, ab_final, aba_final, b_final, ba_final, bab_final)):
        sample_opt = np.concatenate([np.reshape(minmax_normalize(sample), (1, height, width)) for sample in (a, ab, aba, b, ba, bab)], axis=0)
        im.imwrite(im.immerge(sample_opt, 2, 3), '{}/{}.jpg'.format(save_dir, i))
        batching_spectrograms_single.save_reconstructed_audio(a, '{}/a_{}.wav'.format(save_dir, i), iter=500)
        batching_spectrograms_single.save_reconstructed_audio(ab, '{}/ab_{}.wav'.format(save_dir, i), iter=500)
        batching_spectrograms_single.save_reconstructed_audio(aba, '{}/aba_{}.wav'.format(save_dir, i), iter=500)
        batching_spectrograms_single.save_reconstructed_audio(b, '{}/b_{}.wav'.format(save_dir, i), iter=500)
        batching_spectrograms_single.save_reconstructed_audio(ba, '{}/ba_{}.wav'.format(save_dir, i), iter=500)
        batching_spectrograms_single.save_reconstructed_audio(bab, '{}/bab_{}.wav'.format(save_dir, i), iter=500)