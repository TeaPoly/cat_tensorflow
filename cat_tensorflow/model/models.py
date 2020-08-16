#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cat_tensorflow.layer.Convolution2d as conv2d

"""Model definitions.

"""

def sliding_window_batch_with_zeros(inputs,
                                    window_size,
                                    window_stride=1):
    """A sliding window over a dataset.

    Args:
      inputs: List of gradients.
      window_size: The number of elements in the sliding window. It must be positive.
      window_stride:The forward shift of the sliding window in each iteration. The default is 1. It must be positive.

    Returns:
      A sliding window over a dataset.
    """
    with tf.variable_scope('Splice'):
        num_steps = inputs.get_shape()[1].value
        if num_steps is None:
            num_steps = tf.shape(inputs)[1]

        slides = []
        for i in range(window_size):
            one_slide = tf.pad(
                inputs, [[0, 0], [(window_size-1)-i, 0], [0, 0]])
            one_slide = one_slide[:, :num_steps:window_stride, :]

            slides.append(one_slide)
        return tf.concat(slides, axis=2)


def average_gradients(tower_grads):
    """Calculates average gradients.

    Args:
      tower_grads: List of gradients.

    Returns:
      Average gradients.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, v in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def create_model(fingerprint_input, dropout_prob, architecture_parameter, seq_len, is_training, trainable=True):
    """Builds a model of the requested architecture compatible with the settings.

    There are many possible ways of deriving predictions from a spectrogram
    input, so this function provides an abstract interface for creating different
    kinds of models in a black-box way. You need to pass in a TensorFlow node as
    the 'fingerprint' input, and this should output a batch of 1D features that
    describe the audio.

    The function will build the graph it needs in the current TensorFlow graph,
    and return the tensorflow output that will contain the 'logits' input to the
    softmax prediction process. If training flag is on, it will also return a
    placeholder node that can be used to control the dropout amount.

    See the implementations below for the possible model architectures that can be
    requested.

    Args:
      fingerprint_input: tf.placeholder, TensorFlow node that will output audio feature vectors.
      dropout_prob: tf.placeholder, Probability of dropping each neuron.
      architecture_parameter: dict, Dictionary of architecture model parameter.
      fingerprint_size: tf.placeholder, fingerprint sequences length.
      global_mean_std: dict, Dictionary of mean and standard deviation.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.

    Raises:
      Exception: If the architecture type isn't recognized.
    """
    if architecture_parameter['name'] == 'basic_lstm':
        return create_basiclstm(
            fingerprint_input,
            dropout_prob,
            architecture_parameter,
            seq_len,
            is_training,
            trainable)
    else:
        raise Exception('model architecture argument "' + architecture_parameter['name'] +
                        '" not recognized, should be one of "pcen_tdnn_maxpool_fsmn"')


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.

    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)


def create_basiclstm(fingerprint_input, dropout_prob, architecture_parameter, seq_len, is_training, trainable=True):
    """Builds a model with multiple layers.

    Here's the layout of the graph:

        (fingerprint_input)
                 v
             batch-norm
                 v
            Splice/Conv2d
                 v
            BasicLSTM(N)

    Args:
    fingerprint_input: tf.placeholder, TensorFlow node that will output audio feature vectors.
    dropout_prob: tf.placeholder, Probability of dropping each neuron.
    architecture_parameter: dict, Dictionary of architecture model parameter.

    Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

    """
    with tf.variable_scope(architecture_parameter['name'], reuse=tf.AUTO_REUSE):
        if 'batchnorm' in architecture_parameter and architecture_parameter['batchnorm']:
            ###################
            # CMVN
            ###################
            outputs = tf.contrib.layers.batch_norm(fingerprint_input, is_training=is_training, trainable=trainable)
        else:
            outputs = fingerprint_input

        if 'splice' in architecture_parameter:
            ###################
            # Splice
            ###################
            window_size = architecture_parameter['splice']['window_size']
            window_stride_size = architecture_parameter['splice']['window_stride_size']
            seq_len = tf.floor_div(seq_len - 1, window_stride_size) + 1

            tf_windows_size = tf.get_variable(
                name='window_size',
                initializer=window_size,
                trainable=False,
                dtype=tf.int32)

            tf_window_stride_size = tf.get_variable(
                name='window_stride_size',
                initializer=window_stride_size,
                trainable=False,
                dtype=tf.int32)

            outputs = sliding_window_batch_with_zeros(
                outputs, window_size, window_stride_size)\

        if 'conv2d' in architecture_parameter:
            ###################
            # Conv2d
            ###################
            # Expand dimension, convert from (batch_size, seq_len, dimension)
            # to (batch_size, seq_len, dimension, 1)
            outputs = conv2d.Convolution2dExpandDims(outputs)

            param = architecture_parameter['conv2d']
            for i, layer_param in enumerate(param):
                with tf.variable_scope("conv2d_layer%d" % i):
                    outputs, seq_len = conv2d.Convolution2d(
                     outputs,
                     seq_len,
                     filter_sizes=layer_param['filter_size_freq_time'],  # frequency*time, 41x11, 21x11, 21x11
                     num_channels=layer_param['num_channel'],  # 1x32, 32*32, 32*96
                     strides=layer_param['stride_freq_time'],  # frequency*time, 2x2, 2x1, 2x1
                     is_training=is_training,
                     trainable=trainable,
                     keep_dropout_prob=1.0-dropout_prob)

            # Flatten dimension, convert from (batch_size, seq_len, dimension, channel) to
            # (batch_size, seq_len, dimension*channel)
            outputs = conv2d.Convolution2dFlatten(outputs)

        ###################
        # Basic LSTM
        ###################
        if 'lstm' in architecture_parameter:
            param = architecture_parameter['lstm']
            nlayer = len(param['num_units'])

            for i in range(nlayer):
                with tf.variable_scope("basic_lsym_layer%d" % i):
                    cell_fw = tf.nn.rnn_cell.BasicLSTMCell(param['num_units'][i])
                    cell_bw = tf.nn.rnn_cell.BasicLSTMCell(param['num_units'][i])
                    if trainable:
                        cell_fw = tf.contrib.rnn.DropoutWrapper(
                            cell_fw, output_keep_prob=1.0-dropout_prob)
                        cell_bw = tf.contrib.rnn.DropoutWrapper(
                            cell_bw, output_keep_prob=1.0-dropout_prob)
                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=cell_fw, cell_bw=cell_bw, inputs=outputs, dtype=tf.float32, sequence_length=seq_len)
                    outputs = tf.concat(values=outputs, axis=2)

        ##############
        # logits
        ##############
        outputs = tf.layers.dense(outputs, architecture_parameter['label_count'], trainable=trainable, name='logit')

    return outputs, seq_len
