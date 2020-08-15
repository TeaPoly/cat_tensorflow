#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import models  # sliding_window_batch_with_zeros, sliding_window_batch

import logging

logging.getLogger('tensorflow').disabled = True

tf.enable_eager_execution()

if __name__ == '__main__':
  np.random.seed(42)

  # 32,438,258
  batch_size = 32
  fingerprint_size = 522
  fingerprint_dim = 258

  fingerprint_input = np.random.uniform(
      low=0.5, high=13.3, size=(batch_size, fingerprint_size, fingerprint_dim))
  fingerprint_input = fingerprint_input.astype(dtype=np.float32)
  print(fingerprint_input)
  print(fingerprint_input.shape)

  fingerprint_input = tf.convert_to_tensor(fingerprint_input)
  fingerprint_input = models.sliding_window_batch_with_zeros(fingerprint_input,
                                                  3, 1)
  print(fingerprint_input)
  print(fingerprint_input.shape)
