#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import cat_tensorflow.data.input_data as input_data

if __name__ == '__main__':
    model_settings = input_data.prepare_audio_settings(
        16000,
        25, 10, 80)

    data_path = {
            'audio': '/data/203_data/text_database/crf_aishell/test/wav.escp',
            'label': '/data/203_data/text_database/crf_aishell/test/text_number',
            'weight': '/data/203_data/text_database/crf_aishell/test/weight',
            }

    data_batch_size = 6
    data_queue_number = 6
    data_processor_number = 4

    print('data preparing...')

    audio_processor = input_data.AudioProcessor(
        data_path, model_settings)

    print('data finish.')

    audio_processor.start(
        data_batch_size, data_queue_number, data_processor_number, True)

    max_get_data_counts = audio_processor.total_steps
    get_data_counts = 0
    while True:
        start_tick = time.time()
        is_end, data, qsize = audio_processor.get()
        if is_end:
            break
        get_data_counts += 1
        print('(%3d/%3d) duration : #%f queue size : #%d.' %
              (get_data_counts, max_get_data_counts, time.time()-start_tick, qsize))

        start_tick = time.time()
        #  print('\tbatch_size: #{} #weight:{}'.format(batch_size, data[2]))

        #  time.sleep(0.2)
        #  print('\tProcessing duration %f.' % (time.time()-start_tick))
    audio_processor.stop()
