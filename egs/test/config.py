#!/usr/bin/python
# -*- coding: utf-8 -*-

base_parameter = {
    'prop_optimizer': "Adam",  # one of the \'Adam\' or \'RMS\', \'GRAD\'.
    'train_epochs': 1,  # Number of iterations to train'
    'learning_rate_decay_interval_steps': 10,  # 'How many training loops to run
    # How large a learning rate to use when training.
    'learning_rate': 1e-4,
    # How mininum a learning rate to use when training.
    'min_learning_rate': 1e-6,
    # How often to evaluate the training results.
    'eval_step_interval': 10,
    'dropout_prob': 0.5,  # Dropout rate

    # feature
    'sample_rate': 16000,
    # The length of the analysis window in milliseconds.
    'window_size_ms': 25,
    # The step between successive windows in milliseconds.
    'window_stride_ms': 10,
    'fbank_number': 80,  # The number of filters in the filterbank.
    'low_freq': 127,  # Lowest band edge of mel filters. In Hz.
    'high_freq': 7600,  #  highest band edge of mel filters. In Hz.
    'nfft': 512,  # The FFT size.

    # crf
    'crf_lm_path': '',  # Denominator LM path.
    'crf_lamb': -1,  # The weight for the CTC Loss.

}

databuilder_parameter = {
    'train':
        {
            'audio': './data/wav.escp',
            'label': './data/text_number',
            'weight': './data/weight',
            # data augment
            'speed_perb': True,  # Whether speed perturbation is used or not.
        },
    'dev':
        {
            'audio': './data/wav.escp',
            'label': './data/text_number',
            'weight': './data/weight',
            # data augment
            'speed_perb': False,  # Whether speed perturbation is used or not.
        },
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
    #  'lstm':
    #  {
    #  'num_units': [512, 512, 512, 512],
    #  },
    'label_count': 218
}

parameter = {
    'base': base_parameter,
    'data': databuilder_parameter,
    'model': architecture_parameter
}
