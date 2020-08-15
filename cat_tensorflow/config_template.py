#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


base_parameter = {
    'prop_optimizer': "Adam",  # one of the \'Adam\' or \'RMS\', \'GRAD\'.
    'train_epochs': 50,  # Number of iterations to train'
    'learning_rate_decay_interval_steps': 10000,  # 'How many training loops to run
    # How large a learning rate to use when training.
    'learning_rate': 5e-5,
    # How mininum a learning rate to use when training.
    'min_learning_rate': 1e-6,
    # How often to evaluate the training results.
    'eval_step_interval': 2000,
    'dropout_prob': 0.5,  # Dropout rate

    ###############
    # feature
    ###############
    'sample_rate': 16000,
    # The length of the analysis window in milliseconds.
    'window_size_ms': 25,
    # The step between successive windows in milliseconds.
    'window_stride_ms': 10,
    'fbank_number': 80,  # The number of filters in the filterbank.
    'low_freq': 127,  # Lowest band edge of mel filters. In Hz.
    'high_freq': 7600,  # highest band edge of mel filters. In Hz.
    'nfft': 512,  # The FFT size.

    #############
    # crf
    ############
    # Denominator LM path.
    'crf_lm_path': "/ark/repo/ctc_crf/egs/aishell/data/den_meta/den_lm.fst",
    'crf_lamb': 0.01,  # The weight for the CTC Loss.
}

databuilder_parameter = {
    'train':
        {
            'audio': '/data/203_data/text_database/crf_aishell/train/crf_wav.dat',
            'label': '/data/203_data/text_database/crf_aishell/train/crf_lab.dat',
            'weight': '/data/203_data/text_database/crf_aishell/train/crf_weight.dat',
            # data augment
            'speed_perb': True,  # Whether speed perturbation is used or not.
        },
    'dev':
        {
            'audio': '/data/203_data/text_database/crf_aishell/dev/crf_wav.dat',
            'label': '/data/203_data/text_database/crf_aishell/dev/crf_lab.dat',
            'weight': '/data/203_data/text_database/crf_aishell/dev/crf_weight.dat',
            # data augment
            'speed_perb': False,  # Whether speed perturbation is used or not.
        }
}

architecture_parameter = {
    "name": 'basic_lstm',
    'batchnorm': True,
    #  'splice': {'window_size': 5, 'window_stride_size': 3},
    'conv2d': [
        {
            'filter_size_freq_time': (41, 11),
            'num_channel': (1, 32),
            'stride_freq_time': (2, 2)
        },
        {
            'filter_size_freq_time': (21, 11),
            'num_channel': (32, 32),
            'stride_freq_time': (2, 2)
        },
        {
            'filter_size_freq_time': (21, 11),
            'num_channel': (32, 96),
            'stride_freq_time': (2, 1)
        },
    ],
    'lstm':
    {
        'num_units': [320, 320, 320, 320, 320, 320],
    },
    'label_count': 218
}

parameter = {
    'base': base_parameter,
    'data': databuilder_parameter,
    'model': architecture_parameter
}
