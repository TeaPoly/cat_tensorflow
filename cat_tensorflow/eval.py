#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import pickle

import numpy as np
import tensorflow as tf

import cat_tensorflow.data.input_data as input_data
import cat_tensorflow.model.models as models
import cat_tensorflow.utils.import_pyfile as import_pyfile

import logging


os.environ["OMP_NUM_THREADS"] = "1"  # mkl_rt.mkl_set_num_threads(1)

# We want to see all the logging messages for this tutorial.
tf.logging.set_verbosity(tf.logging.ERROR)

def PrintSetting(logger, parameter, title=''):
    logger.info('%s:' % (title))
    for key, val in parameter.items():
        logger.info('\t\'{}\': \'{}\''.format(key, val))


def main(_):
    # Logging writer setting.
    logger = logging.getLogger('CAT_TENSROFLOW')
    logger.setLevel(logging.DEBUG)

    # GPU settings
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_settings
    config_proto = tf.ConfigProto()  # log_device_placement=True
    config_proto.gpu_options.allow_growth = True
    config_proto.allow_soft_placement = True

    if not os.path.exists(FLAGS.save_logit_dir):
        os.makedirs(FLAGS.save_logit_dir)

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession(config=config_proto)

    work_space = os.path.dirname(FLAGS.start_checkpoint)

    parameter_handler = pickle.load(open(os.path.join(work_space, '../parameter.pkl'), 'rb'))

    # Load architecture parameter dictionary.
    base_parameter = parameter_handler['base']
    architecture_parameter = parameter_handler['model']
    model_settings = input_data.prepare_audio_settings(base_parameter)

    # Print architecture parameter.
    PrintSetting(logger, vars(FLAGS), 'FLAGS')
    PrintSetting(logger, base_parameter, 'Base setting')
    PrintSetting(logger, architecture_parameter, 'Acoustic model setting')
    PrintSetting(logger, model_settings, 'Feature setting')

    # Bulding graph.
    logger.debug('Building graph...')
    fingerprint_dim = model_settings['nfilt']
    fingerprint_input = tf.placeholder(
        tf.float32, [FLAGS.batch_size, None, fingerprint_dim], name='fingerprint_input')
    seq_len = tf.placeholder(
        tf.int32, [FLAGS.batch_size], name='seq_len')
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    is_training = tf.placeholder(tf.bool, name="is_training")

    # GPU parallel.
    GPU_devices = FLAGS.GPU_settings.split(',')
    GPU_number = len(GPU_devices)

    # Data paraller
    multi_feature_list = [fingerprint_input]
    multi_seq_len_list = [seq_len]
    if GPU_number >= 2:
        multi_feature_list = tf.split(fingerprint_input, GPU_number, 0)
        multi_seq_len_list = tf.split(seq_len, GPU_number, 0)

    multi_gpu_logits = []
    multi_gpu_seq_lens = []
    for GPU_index in range(GPU_number):
        part_feature = multi_feature_list[GPU_index]
        part_seq_len = multi_seq_len_list[GPU_index]

        GPU_device_name = "/device:GPU:{}".format(GPU_index)
        logger.debug("\t Building graph in \"{}\"".format(GPU_device_name))

        # Create a device context.
        with tf.device(GPU_device_name):
            with tf.name_scope("cat_tensorflow_gpu{}".format(GPU_index)):
                part_logits, part_seq_len = models.create_model(
                    part_feature,
                    dropout_prob,
                    architecture_parameter,
                    part_seq_len,
                    is_training=is_training,
                    trainable=False)
                multi_gpu_logits.append(part_logits)
                multi_gpu_seq_lens.append(part_seq_len)

    final_logits = tf.concat(multi_gpu_logits, axis=0)
    final_seq_len = tf.concat(multi_gpu_seq_lens, axis=0)

    tf.global_variables_initializer().run()
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    logger.info('Training from checkpoint: %s ',
                FLAGS.start_checkpoint)

    #######################################
    ##           Eval                 ##
    #######################################
    testing_audio_processor = input_data.AudioProcessor(
        FLAGS.audio_scp, model_settings)

    test_set_size = testing_audio_processor.size

    testing_audio_processor.start(FLAGS.batch_size,
                                  FLAGS.data_queue_number,
                                  FLAGS.data_processor_number)

    uuid_logitpath = {}
    test_interval_steps = 0
    while True:
        is_end, data, qsize = testing_audio_processor.get()
        if is_end:
            break

        # Run a test step and capture training summaries for TensorBoard
        # with the `merged` op.
        logits, sequence_lens = sess.run(
            [final_logits, final_seq_len],
            feed_dict={
                fingerprint_input: data['feature'],
                seq_len: data['seq_len'],
                is_training: False,
                dropout_prob: 0.0,
            })

        for i, uuid in enumerate(data['uuid']):
            sequence_len = sequence_lens[i]
            logit = logits[i,:sequence_len,:]
            save_logit_path = os.path.join(FLAGS.save_logit_dir, uuid)
            np.save(save_logit_path, logit)
            uuid_logitpath[uuid] = save_logit_path

        test_interval_steps += 1

        logger.info('\tstep : %d/%d' %
                        (test_interval_steps, testing_audio_processor.total_steps))

    testing_audio_processor.stop()

    with open(FLAGS.logit_scp, 'w') as fp:
        for uuid, logitpath in uuid_logitpath.items():
            fp.write("{} {}\n".format(uuid, logitpath+'.npy'))
        logger.info('Logit result has written in \"{}\"'.format(FLAGS.logit_scp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='How many items to train with at once',)
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--data_queue_number',
        type=int,
        default=32,
        help='Collect data queue number')
    parser.add_argument(
        '--GPU_settings',
        type=str,
        default="0,1",
        help='Using GPUs device name (i.e. 0,1).')
    parser.add_argument(
        '--data_processor_number',
        type=int,
        default=4,
        help="""\
      Number of multi-process for processing data.
      """)
    parser.add_argument(
        '--audio_scp',
        type=str,
        default="",
        help="""\
        The script file include uuid and wave path.
      """)
    parser.add_argument(
        '--save_logit_dir',
        type=str,
        default='',
        help='Directory to save logit.',)
    parser.add_argument(
        '--logit_scp',
        type=str,
        default='',
        help='The script file to save uuid logit path.',)

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
