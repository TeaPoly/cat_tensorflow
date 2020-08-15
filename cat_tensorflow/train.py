#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#  import importlib
import datetime
import argparse
import os
import sys
import time

import pickle

import numpy as np
import tensorflow as tf

import cat_tensorflow.data.input_data as input_data
import cat_tensorflow.model.models as models
import cat_tensorflow.utils.import_pyfile as import_pyfile

#  import sys_utils

import ctc_crf_tensorflow

import logging

from tensorflow.python.util import deprecation_wrapper as deprecation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
deprecation._PRINT_DEPRECATION_WARNINGS = False
deprecation._PER_MODULE_WARNING_LIMIT = 0

os.environ["OMP_NUM_THREADS"] = "1"  # mkl_rt.mkl_set_num_threads(1)


def PrintSetting(logger, parameter, title=''):
    logger.info('%s:' % (title))
    for key, val in parameter.items():
        logger.info('\t\'{}\': \'{}\''.format(key, val))


def main(_):
    # GPU settings
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_settings
    config_proto = tf.ConfigProto()  # log_device_placement=True
    config_proto.gpu_options.allow_growth = True
    config_proto.allow_soft_placement = True

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession(config=config_proto)

    # Make dir for training folder
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)

    logger_dir = os.path.join(FLAGS.train_dir, 'logger')
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)
    model_dir = os.path.join(FLAGS.train_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    summaries_dir = os.path.join(FLAGS.train_dir, 'summary')
    if not os.path.exists(summaries_dir):
        os.makedirs(summaries_dir)

    # Load mean std dictionary.
    #  global_mean_std = importlib.import_module(
        #  'parameter.{}'.format(FLAGS.global_mean_std_file)).global_mean_std

    # Saving global mean std data for loading model.
    #  global_mean_std_path = os.path.join(
        #  FLAGS.train_dir, 'meanstd.pkl')
    #  with open(global_mean_std_path, 'wb') as fp:
        #  pickle.dump(global_mean_std, fp)
    #  tf.logging.info('Saving global mean and std to "%s"', global_mean_std_path)

    # Print meanstd.
    #  tf.logging.info('Global Mean Std: ')
    #  for key, val in global_mean_std.items():
        #  tf.logging.info('\t\'{}\': \'{}\''.format(key, val))

    # Logging writer setting.
    logger = logging.getLogger('CAT_TENSORFLOW')
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s - %(name)s - %(levelname)s]: %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    now = datetime.datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    logging_path = os.path.join(
        logger_dir,
        '{}.log'.format(now))
    handler = logging.FileHandler(logging_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Load architecture parameter dictionary.
    parameter_handler = import_pyfile.import_pyfile(
        FLAGS.parameter, 'parameter').parameter

    # Saving parameter for eval.
    parameter_path = os.path.join(
        FLAGS.train_dir, 'parameter.pkl')
    with open(parameter_path, 'wb') as fp:
        pickle.dump(parameter_handler, fp)
    logger.info('Saving architecture parameter to "%s"', parameter_path)

    # Loading different module settings
    base_parameter = parameter_handler['base']
    databuilder_parameter = parameter_handler['data']
    architecture_parameter = parameter_handler['model']
    model_settings = input_data.prepare_audio_settings(base_parameter)

    # Print settings.
    PrintSetting(logger, vars(FLAGS), 'FLAGS')
    PrintSetting(logger, base_parameter, 'Base setting')
    PrintSetting(logger, databuilder_parameter, 'Data builder setting')
    PrintSetting(logger, architecture_parameter, 'Acoustic model setting')
    PrintSetting(logger, model_settings, 'Feature setting')

    # Bulding graph.
    logger.debug('Building graph...')
    fingerprint_dim = model_settings['nfilt']
    fingerprint_input = tf.placeholder(
        tf.float32, [FLAGS.batch_size, None, fingerprint_dim], name='fingerprint_input')
    seq_len = tf.placeholder(
        tf.int32, [FLAGS.batch_size], name='seq_len')
    label = tf.sparse_placeholder(tf.int32)
    weight = tf.placeholder(
        tf.float32, [FLAGS.batch_size], name='weight')
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    is_training = tf.placeholder(tf.bool, name="is_training")

    # Optionally we can add runtime checks to spot when NaNs or other symptoms of
    # numerical errors start occurring during training.
    control_dependencies = []
    if FLAGS.check_nans:
        checks = tf.add_check_numerics_ops()
        control_dependencies = [checks]

    with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
        prop_optimizer = base_parameter['prop_optimizer']
        learning_rate_input = tf.placeholder(
            tf.float32, [], name='learning_rate_input')
        if prop_optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate_input)
        elif prop_optimizer == 'RMS':
            optimizer = tf.train.RMSPropOptimizer(learning_rate_input)
        elif prop_optimizer == 'GRAD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate_input)
        else:
            raise ValueError(
                'prop_optimizer should be one of \'Adam\',\'RMS\',\'GRAD\'.')

    # GPU parallel.
    GPU_devices = FLAGS.GPU_settings.split(',')
    GPU_number = len(GPU_devices)
    crf_lamb = base_parameter['crf_lamb']
    if crf_lamb > 0:
        assert GPU_number > 0
        gpus = list(range(GPU_number))
        crf_lm_path = base_parameter['crf_lm_path']
        if not os.path.isfile(crf_lm_path):
            raise Exception(
                'CRF LM path {} setting error.'.format(crf_lm_path))
        ctc_crf_init_ops = ctc_crf_tensorflow.ctc_crf_init_env(
            crf_lm_path, gpus)
        ctc_crf_release_ops = ctc_crf_tensorflow.ctc_crf_release_env(
            gpus)

    # Data paraller
    multi_weight_list = [weight]
    multi_label_list = [label]
    multi_feature_list = [fingerprint_input]
    multi_seq_len_list = [seq_len]
    if GPU_number >= 2:
        multi_weight_list = tf.split(weight, GPU_number, 0)
        multi_label_list = tf.sparse_split(
            sp_input=label, num_split=GPU_number, axis=0)
        multi_feature_list = tf.split(fingerprint_input, GPU_number, 0)
        multi_seq_len_list = tf.split(seq_len, GPU_number, 0)

    multi_gpu_losses = []
    #  multi_gpu_ters = []
    tower_grads = []
    for GPU_index in range(GPU_number):
        part_weight = multi_weight_list[GPU_index]
        part_label = multi_label_list[GPU_index]
        part_feature = multi_feature_list[GPU_index]
        part_seq_len = multi_seq_len_list[GPU_index]

        GPU_device_name = "/device:GPU:{}".format(GPU_index)
        logger.debug("\t Building graph in \"{}\"".format(GPU_device_name))

        # Create a device context.
        with tf.device(GPU_device_name):
            with tf.name_scope("cat_tensorflow_gpu{}".format(GPU_index)):
                logits, part_seq_len = models.create_model(
                    part_feature,
                    dropout_prob,
                    architecture_parameter,
                    part_seq_len,
                    is_training=is_training,
                    trainable=True)

                with tf.name_scope('loss'):
                    # Convert to time major
                    logits_timemajor = tf.transpose(logits, (1, 0, 2))
                    loss = ctc_crf_tensorflow.ctc_crf_loss(
                        tf.nn.log_softmax(logits_timemajor),
                        labels=part_label,
                        input_lengths=part_seq_len,
                        blank_label=0,
                        lamb=crf_lamb)

                #  with tf.name_scope('ter'):
                    #  # Option 2: tf.nn.ctc_beam_search_decoder
                    #  # (it's slower but you'll get better results)
                    #  decoded, log_prob = tf.nn.ctc_greedy_decoder(
                    #  tf.roll(logits_timemajor, shift=-1, axis=2), part_seq_len)

                    #  # Inaccuracy: label error rate
                    #  best_decoded = decoded[0]
                    #  best_decoded = tf.sparse.SparseTensor(
                    #  best_decoded.indices,
                    #  best_decoded.values+1, best_decoded.dense_shape)
                    #  ter = tf.reduce_mean(tf.edit_distance(tf.cast(best_decoded, tf.int32),
                    #  part_label, normalize=False))

                grads = optimizer.compute_gradients(loss)

                if crf_lamb > 0:
                    loss = loss - tf.reduce_mean(part_weight)

                multi_gpu_losses.append(loss)
                #  multi_gpu_ters.append(ter)
                tower_grads.append(grads)

    tower_grads = models.average_gradients(tower_grads)
    train_step = optimizer.apply_gradients(tower_grads)

    loss = tf.reduce_mean(multi_gpu_losses)
    #  ter = tf.reduce_mean(multi_gpu_ters)

    # Summarize losses on all devices
    with tf.get_default_graph().name_scope('eval'):
        tf.summary.scalar('loss', loss)
        #  tf.summary.scalar('ter', ter)

    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)

    saver = tf.train.Saver(tf.global_variables(),
                           max_to_keep=FLAGS.max_to_keep)

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged_summaries = tf.summary.merge_all(scope='eval')
    train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                         sess.graph)
    valid_writer = tf.summary.FileWriter(
        summaries_dir + '/validation')

    tf.global_variables_initializer().run()

    # Logging for best checkpoint
    train_total_steps = 0

    cont_checkpoint = ''
    best_path = os.path.join(FLAGS.train_dir, 'best.txt')
    if FLAGS.start_checkpoint:
        cont_checkpoint = FLAGS.start_checkpoint
    elif os.path.isfile(best_path):
        with open(best_path, 'r') as fp:
            best_checkpoint = fp.readline().strip()
            if best_checkpoint:
                cont_checkpoint = best_checkpoint

    if cont_checkpoint:
        models.load_variables_from_checkpoint(sess, cont_checkpoint)
        train_total_steps = global_step.eval(session=sess)
        logger.info('Training from checkpoint: %s, steps: %d ',
                    cont_checkpoint, train_total_steps)

    logger.debug('Training from steps: %d ', train_total_steps)
    init_global_steps = train_total_steps

    #######################################
    training_audio_processor = input_data.AudioProcessor(
        databuilder_parameter['train'], model_settings)

    validation_audio_processor = input_data.AudioProcessor(
        databuilder_parameter['dev'], model_settings)

    '''
    variables_names = [v.name for v in tf.trainable_variables()]
    logger.info('Trainable variables names: ')
    for variables_name in variables_names:
        logger.info('\t{}'.format(variables_name))
    exit(0)
    '''

    # Save graph.pbtxt.
    tf.train.write_graph(sess.graph_def, model_dir,
                         'model.pbtxt')

    # Init crf ops
    if crf_lamb > 0:
        sess.run(ctc_crf_init_ops)

    # Training loop.
    logger.debug('Training epoch starting...')
    learning_rate_value = base_parameter['learning_rate']
    epochs = base_parameter['train_epochs']
    best_valid_loss = sys.float_info.max
    try:
        for epoch in range(1, epochs+1):
            train_set_size = training_audio_processor.size

            timestamp = datetime.datetime.now()
            #  logger.warning('===> %s: Epoch #%d, Training start (N=%d).' %
            #  (timestamp, epoch, train_set_size))

            training_audio_processor.start(FLAGS.batch_size,
                                           FLAGS.data_queue_number,
                                           FLAGS.data_processor_number)

            max_steps = training_audio_processor.total_steps

            train_final_loss = 0
            train_interval_steps = 0
            while True:
                tick_start = time.time()
                # logger.info('\t >> Before get data, qsize: %d' % training_audio_processor.qsize)
                is_end, data, qsize = training_audio_processor.get()
                cos_duration = time.time() - tick_start
                #  logger.debug('\tGet data duration: %f, qsize: %d' %
                #  (cos_duration, qsize))
                if is_end:
                    break

                if (train_total_steps != 0) and (train_total_steps % base_parameter['learning_rate_decay_interval_steps'] == 0) and (learning_rate_value > base_parameter['min_learning_rate']):
                    learning_rate_value = learning_rate_value * 0.5

                tick_start = time.time()
                # Run the graph with this batch of training data.
                train_summary, train_step_loss, _, train_total_steps = sess.run(
                    [
                        merged_summaries, loss, train_step, increment_global_step
                    ],
                    feed_dict={
                        fingerprint_input: data['feature'],
                        label: data['label'],
                        seq_len: data['seq_len'],
                        weight: data['weight'],
                        is_training: True,
                        learning_rate_input: learning_rate_value,
                        dropout_prob: base_parameter['dropout_prob'],
                    })
                cos_duration = time.time() - tick_start

                # Superimposed counter.
                train_interval_steps += 1

                logger.debug('Epochs-Steps: #%d-%d, progress (%d/%d), learing rate: %f, training loss: %.6e, duration: %f' %
                             (epoch, train_total_steps, train_interval_steps, training_audio_processor.total_steps,
                                 learning_rate_value, train_step_loss, cos_duration))

                eval_step_interval = base_parameter['eval_step_interval']
                train_final_loss += train_step_loss / eval_step_interval

                train_writer.add_summary(train_summary, train_total_steps)

                is_last_step = ((train_total_steps-init_global_steps)
                                == epochs*max_steps)
                if is_last_step or (eval_step_interval > 0 and (train_total_steps % eval_step_interval) == 0):
                    #######################################
                    ##           validation              ##
                    #######################################
                    tick_start = time.time()

                    valid_set_size = validation_audio_processor.size

                    validation_audio_processor.start(FLAGS.batch_size,
                                                     FLAGS.data_queue_number,
                                                     FLAGS.data_processor_number,
                                                     True)

                    valid_final_loss = 0
                    valid_interval_steps = 0
                    while True:
                        is_end, data, qsize = validation_audio_processor.get()
                        if is_end:
                            break

                        # Run a validation step and capture training summaries for TensorBoard
                        # with the `merged` op.
                        valid_summary, valid_step_loss = sess.run(
                            [merged_summaries, loss],
                            feed_dict={
                                fingerprint_input: data['feature'],
                                label: data['label'],
                                seq_len: data['seq_len'],
                                weight: data['weight'],
                                is_training: False,
                                learning_rate_input: learning_rate_value,
                                dropout_prob: 0.0,
                            })

                        valid_interval_steps += 1

                        valid_writer.add_summary(
                            valid_summary, train_total_steps)

                        valid_final_loss += (valid_step_loss *
                                             FLAGS.batch_size) / valid_set_size

                        logger.debug('\tvalid step : %d/%d, loss: %.6e' %
                                     (valid_interval_steps, validation_audio_processor.total_steps, valid_step_loss))

                    validation_audio_processor.stop()

                    cos_duration = time.time() - tick_start
                    logger.debug('validation time consuming %f.' %
                                 (cos_duration))

                    timestamp = datetime.datetime.now()

                    # training logger
                    logging_string = '%s: train step: #%d, loss: %.6e' % (
                        timestamp, train_total_steps, train_final_loss)
                    train_final_loss = 0
                    logger.info(logging_string)

                    # validation logger
                    logging_string = '%s: valid step: #%d, loss: %.6e (N=%d)' % (
                        timestamp, train_total_steps, valid_final_loss, valid_set_size)

                    logger.info(logging_string)

                    # Save the model checkpoint periodically.
                    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
                    logger.info('Saving to "%s-%d"',
                                checkpoint_path, train_total_steps)

                    if valid_final_loss < best_valid_loss:
                        with open(best_path, 'w') as fp:
                            fp.write('{}-{}\n'.format(
                                checkpoint_path, train_total_steps))
                            fp.write('loss: {}\n'.format(valid_final_loss))

                    saver.save(sess, checkpoint_path,
                               global_step=train_total_steps)

            training_audio_processor.stop()
            timestamp = datetime.datetime.now()
            logger.debug('===> %s: Epoch #%d, Training end.' %
                         (timestamp, epoch))
    except KeyboardInterrupt:
        # Save the model checkpoint periodically.
        checkpoint_path = os.path.join(model_dir, 'model_interrupt' + '.ckpt')
        saver.save(sess, checkpoint_path, global_step=train_total_steps)
        logger.info('Saving to "%s-%d"',
                    checkpoint_path, train_total_steps)

    if crf_lamb > 0:
        sess.run(ctc_crf_release_ops)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='How many items to train with at once',)
    parser.add_argument(
        '--train_dir',
        type=str,
        default='../train_logs',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--parameter',
        type=str,
        default='',
        help='Parameter path.')
    #  parser.add_argument(
    #  '--global_mean_std_file',
    #  type=str,
    #  default='8k_mean_std',
    #  help='Global mean std file.')
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=False,
        help='Whether to check for invalid numbers during processing')
    parser.add_argument(
        '--data_queue_number',
        type=int,
        default=32,
        help='Collect data queue number')
    parser.add_argument(
        '--GPU_settings',
        type=str,
        default="",
        help='Using GPUs device name (i.e. 0,1).')
    parser.add_argument(
        '--max_to_keep',
        type=int,
        default=10,
        help='Maximum number of recent checkpoints to keep.')
    parser.add_argument(
        '--data_processor_number',
        type=int,
        default=2,
        help="""\
      Number of multi-process for processing data.
      """)

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
