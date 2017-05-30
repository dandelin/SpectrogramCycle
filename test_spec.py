from __future__ import absolute_import, division, print_function

import os
import batching_spectrograms
import utils
import models_spec
import argparse
import numpy as np
import tensorflow as tf
import image_utils as im

from glob import glob


""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', dest='dataset', default='horse2zebra', help='which dataset to use')
args = parser.parse_args()

dataset = args.dataset
a_test_path, b_test_path = ['./datasets/' + dataset + '/test{}/'.format(e) for e in ['A', 'B']]

height, width = batching_spectrograms.wh_from_sample(a_train_path)


""" run """
with tf.Session() as sess:
    a_real = tf.placeholder(tf.float32, shape=[None, height, width, 1])
    b_real = tf.placeholder(tf.float32, shape=[None, height, width, 1])

    a2b = models_spec.generator(a_real, 'a2b')
    b2a = models_spec.generator(b_real, 'b2a')
    b2a2b = models_spec.generator(b2a, 'a2b', reuse=True)
    a2b2a = models_spec.generator(a2b, 'b2a', reuse=True)

    # restore
    saver = tf.train.Saver()
    ckpt_path = utils.load_checkpoint('./checkpoints/' + dataset, sess, saver)
    if ckpt_path is None:
        raise Exception('No checkpoint!')
    else:
        print('Copy variables from % s' % ckpt_path)

    # test

    save_dir = './test_predictions/' + dataset + '/'
    utils.mkdir([save_dir])

    a_test_pool = batching_spectrograms.SpecData(sess, a_test_path, batch_size, retain_all=True)
    b_test_pool = batching_spectrograms.SpecData(sess, b_test_path, batch_size, retain_all=True)
    a_final = []
    ab_final = []
    aba_final = []
    b_final = []
    ba_final = []
    bab_final = []
    for i in range(len(a_test_pool)):
        a_real_ipt = a_test_pool.batch()
        a2b_opt, a2b2a_opt = sess.run([a2b, a2b2a], feed_dict={a_real: a_real_ipt})
        ab_final.append(a2b_opt)
        aba_final.append(a2b2a_opt)
        a_final.append(a_real_ipt)

    for i in range(len(b_list)):
        b_real_ipt = b_test_pool.batch()
        b2a_opt, b2a2b_opt = sess.run([b2a, b2a2b], feed_dict={b_real: b_real_ipt})
        ba_final.append(b2a_opt)
        bab_final.append(b2a2b_opt)
        b_final.append(b_real_ipt)
    
    a = np.concatenate(a_final, axis=2)
    ab = np.concatenate(ab_final, axis=2)
    aba = np.concatenate(aba_final, axis=2)
    b = np.concatenate(b_final, axis=2)
    ba = np.concatenate(ba_final, axis=2)
    bab = np.concatenate(baa_final, axis=2)

    for name, e in zip(['a', 'ab', 'aba', 'b', 'ba', 'bab'], [a, ab, aba, b, ba, bab]):
        batching_spectrograms.save_reconstructed_audio(e, '{}/{}.wav'.format(save_dir, name), iter=500)