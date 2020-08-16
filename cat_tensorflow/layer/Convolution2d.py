#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

'''
Architecture Channels Filter dimension Stride Regular Dev Noisy Dev
    3-layer 2D 32, 32, 96 41x11, 21x11, 21x11 2x2, 2x1, 2x1 8.61 14.74
    For 2D-invariant convolutions the first dimension
    is frequency and the second dimension is time. All models have BatchNorm, SortaGrad, and 35 million
'''
def Convolution2d(
                 inputs,
                 seq_len,
                 filter_sizes=(41, 11),  # frequency*time, 41x11, 21x11, 21x11
                 num_channels=(1, 32),  # 1x32, 32*32, 32*96
                 strides=(2, 2),  # frequency*time, 2x2, 2x1, 2x1
                 is_training=False,
                 trainable=True,
                 keep_dropout_prob=1.0):
    freq_stride, time_stride = strides
    freq_filter_size, time_filter_size = filter_sizes
    num_channels_in, num_channels_out = num_channels

    # Convolution2d.
    conv2d_filter = tf.get_variable(
        shape=[time_filter_size, freq_filter_size,
               num_channels_in, num_channels_out],
        initializer=tf.contrib.layers.xavier_initializer(),
        name="conv2d_filter",
        trainable=trainable)
    # NHWC: [filter_height, filter_width, in_channels, out_channels]
    net = tf.nn.conv2d(
        input=inputs,
        strides=[1, time_stride, freq_stride, 1],
        filter=conv2d_filter,
        padding='SAME')  # NHWC: [batch size, time-domain, frequency-domain, output channels]

    # Nonlinearity.
    net = tf.nn.relu(net)

    # Dropout
    #  if keep_dropout_prob != 1.0:
        #  net = tf.cond(keep_dropout_prob < 1.0,
                      #  lambda: tf.nn.dropout(
                          #  net, rate=1.0-keep_dropout_prob),
                      #  lambda: net)

    # Batch Normalization.
    net = tf.contrib.layers.batch_norm(net, is_training=is_training, trainable=trainable)

    seq_len = tf.cast(
        tf.ceil(seq_len/time_stride), tf.int32)

    return net, seq_len

def Convolution2dExpandDims(inputs):
    '''
    N,H,(C*W)->N,H,W,C
    '''
    return tf.expand_dims(inputs, -1)  # [batch, time, frequency, 1]


def Convolution2dFlatten(inputs):
    '''
    N,H,W,C->N,H,(C*W)
    '''
    shape = inputs.get_shape()
    return tf.reshape(inputs, [shape[0], -1, shape[2]*shape[3]])
