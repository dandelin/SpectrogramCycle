from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ops
import batching_spectrograms_single
import utils
import models_spec
import argparse
import numpy as np
import tensorflow as tf
import image_utils as im
from scipy import stats

from glob import glob


""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', dest='dataset', default='speech3', help='which dataset to use')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in a batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='GPU ID')
args = parser.parse_args()

dataset = args.dataset
epoch = args.epoch
batch_size = args.batch_size
lr = args.lr
gpu_id = args.gpu_id

a_train_path, b_train_path = ['./datasets/' + dataset + '/train{}/'.format(e) for e in ['A', 'B']]
a_test_path, b_test_path = ['./datasets/' + dataset + '/test{}/'.format(e) for e in ['A', 'B']]

height, width = 512, 512

""" graphs """
with tf.device('/gpu:%d' % gpu_id):
    ''' graph '''
    # nodes
    a_real = tf.placeholder(tf.float32, shape=[None, height, width, 1])
    b_real = tf.placeholder(tf.float32, shape=[None, height, width, 1])
    a2b_sample = tf.placeholder(tf.float32, shape=[None, height, width, 1])
    b2a_sample = tf.placeholder(tf.float32, shape=[None, height, width, 1])

    a2b = models_spec.generator(a_real, 'a2b')
    b2a = models_spec.generator(b_real, 'b2a')
    b2a2b = models_spec.generator(b2a, 'a2b', reuse=True)
    a2b2a = models_spec.generator(a2b, 'b2a', reuse=True)

    a_dis = models_spec.discriminator(a_real, 'a')
    b2a_dis = models_spec.discriminator(b2a, 'a', reuse=True)
    b2a_sample_dis = models_spec.discriminator(b2a_sample, 'a', reuse=True)
    b_dis = models_spec.discriminator(b_real, 'b')
    a2b_dis = models_spec.discriminator(a2b, 'b', reuse=True)
    a2b_sample_dis = models_spec.discriminator(a2b_sample, 'b', reuse=True)

    # losses
    g_loss_a2b = tf.identity(ops.l2_loss(a2b_dis, tf.ones_like(a2b_dis)), name='g_loss_a2b')
    g_loss_b2a = tf.identity(ops.l2_loss(b2a_dis, tf.ones_like(b2a_dis)), name='g_loss_b2a')
    cyc_loss_a = tf.identity(ops.l1_loss(a_real, a2b2a) * 10.0, name='cyc_loss_a')
    cyc_loss_b = tf.identity(ops.l1_loss(b_real, b2a2b) * 10.0, name='cyc_loss_b')
    g_loss = g_loss_a2b + g_loss_b2a + cyc_loss_a + cyc_loss_b

    d_loss_a_real = ops.l2_loss(a_dis, tf.ones_like(a_dis))
    d_loss_b2a_sample = ops.l2_loss(b2a_sample_dis, tf.zeros_like(b2a_sample_dis))
    d_loss_a = tf.identity((d_loss_a_real + d_loss_b2a_sample) / 2.0, name='d_loss_a')
    d_loss_b_real = ops.l2_loss(b_dis, tf.ones_like(b_dis))
    d_loss_a2b_sample = ops.l2_loss(a2b_sample_dis, tf.zeros_like(a2b_sample_dis))
    d_loss_b = tf.identity((d_loss_b_real + d_loss_a2b_sample) / 2.0, name='d_loss_b')

    # summaries
    g_summary = ops.summary_tensors([g_loss_a2b, g_loss_b2a, cyc_loss_a, cyc_loss_b])
    d_summary_a = ops.summary(d_loss_a)
    d_summary_b = ops.summary(d_loss_b)

    ''' optim '''
    t_var = tf.trainable_variables()
    d_a_var = [var for var in t_var if 'a_discriminator' in var.name]
    d_b_var = [var for var in t_var if 'b_discriminator' in var.name]
    g_var = [var for var in t_var if 'a2b_generator' in var.name or 'b2a_generator' in var.name]

    d_a_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss_a, var_list=d_a_var)
    d_b_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss_b, var_list=d_b_var)
    g_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_var)


""" train """
''' init '''
# session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# counter
it_cnt, update_cnt = ops.counter()

'''data'''
a_data_pool = batching_spectrograms_single.SpecData(sess, a_train_path, batch_size)
b_data_pool = batching_spectrograms_single.SpecData(sess, b_train_path, batch_size)

a_test_pool = batching_spectrograms_single.SpecData(sess, a_test_path, batch_size)
b_test_pool = batching_spectrograms_single.SpecData(sess, b_test_path, batch_size)

a2b_pool = utils.ItemPool()
b2a_pool = utils.ItemPool()

'''summary'''
summary_writer = tf.summary.FileWriter('./summaries/' + dataset, sess.graph)

'''saver'''
ckpt_dir = './checkpoints/' + dataset
utils.mkdir(ckpt_dir + '/')

saver = tf.train.Saver(max_to_keep=5)
ckpt_path = utils.load_checkpoint(ckpt_dir, sess, saver)
if ckpt_path is None:
    sess.run(tf.global_variables_initializer())
else:
    print('Copy variables from % s' % ckpt_path)

'''train'''
try:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    batch_epoch = min(len(a_data_pool), len(b_data_pool)) // batch_size
    max_it = epoch * batch_epoch
    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)

        # prepare data
        a_real_ipt = a_data_pool.batch()
        b_real_ipt = b_data_pool.batch()
        a2b_opt, b2a_opt = sess.run([a2b, b2a], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
        a2b_sample_ipt = np.array(a2b_pool(list(a2b_opt)))
        b2a_sample_ipt = np.array(b2a_pool(list(b2a_opt)))

        stat = [stats.describe(arr, axis=None) for arr in [a_real_ipt, b_real_ipt, a2b_opt, b2a_opt, a2b_sample_ipt, b2a_sample_ipt]]
        for s in stat:
            print(s)

        # train G
        g_summary_opt, _ = sess.run([g_summary, g_train_op], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
        summary_writer.add_summary(g_summary_opt, it)
        # train D_b
        d_summary_b_opt, _ = sess.run([d_summary_b, d_b_train_op], feed_dict={b_real: b_real_ipt, a2b_sample: a2b_sample_ipt})
        summary_writer.add_summary(d_summary_b_opt, it)
        # train D_a
        d_summary_a_opt, _ = sess.run([d_summary_a, d_a_train_op], feed_dict={a_real: a_real_ipt, b2a_sample: b2a_sample_ipt})
        summary_writer.add_summary(d_summary_a_opt, it)

        # which epoch
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        # display
        if it % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        # save
        if (it + 1) % 1000 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        if (it) % 500 == 0:
            a_real_ipt = a_test_pool.batch()
            b_real_ipt = b_test_pool.batch()
            [a2b_opt, a2b2a_opt, b2a_opt, b2a2b_opt] = sess.run([a2b, a2b2a, b2a, b2a2b], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
            samples = [a_real_ipt, a2b_opt, a2b2a_opt, b_real_ipt, b2a_opt, b2a2b_opt]
            sample_opt = np.concatenate([np.reshape(sample, (1, height, width)) for sample in samples], axis=0)

            save_dir = './sample_images_while_training/' + dataset
            utils.mkdir(save_dir + '/')
            im.imwrite(im.immerge(sample_opt, 2, 3), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))
            for i, (name, sample) in enumerate(zip(['a', 'ab', 'aba', 'b', 'ba', 'bab'], samples)):
                batching_spectrograms_single.save_reconstructed_audio(sample, '{}/Epoch{}_{}of{}_{}.wav'.format(save_dir, epoch, it_epoch, batch_epoch, name))

except Exception as e:
    coord.request_stop(e)
finally:
    print("Stop threads and close session!")
    coord.request_stop()
    coord.join(threads)
    sess.close()
