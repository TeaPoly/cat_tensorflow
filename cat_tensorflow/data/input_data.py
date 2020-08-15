#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import math

import cat_tensorflow.data.utility as utils
from cat_tensorflow.data.audio import AudioSegment
import python_speech_features

import numpy as np
from tqdm import tqdm

RANDOM_SEED = 42


def convert_sparse_label(label_list, batch_size, label_len_list):
    """ Convert label list to sparse label for CTC loss.
    """
    labels_idx = []
    labels_val = []
    for i, labels in enumerate(label_list):
        for j, label in enumerate(labels):
            labels_idx.append([i, j])
            labels_val.append(label)

    yval_np = np.asarray(labels_val, dtype=np.int32)
    yidx_np = np.asarray(labels_idx, dtype=np.int32)
    yshape_np = np.array(
        [batch_size, max(label_len_list)], dtype=np.int32)

    return ((yidx_np, yval_np, yshape_np))


def prepare_audio_settings(parameters):
    """Calculates common settings needed for all models.

    Args:
      sample_rate: Number of audio samples per second.
      window_size_ms: Duration of frequency analysis window.
      window_stride_ms: How far to move in time between frequency windows.
      mfbank_coefficient_count: Number of frequency bins to use for analysis.

    Returns:
      Dictionary containing common settings.
    """
    window_size_sec = (parameters['window_size_ms'] / 1000.0)
    window_stride_sec = (parameters['window_stride_ms'] / 1000.0)

    return {
        "samplerate": parameters['sample_rate'],
        "nfilt": parameters['fbank_number'],
        "winlen": window_size_sec,
        "winstep": window_stride_sec,
        "nfft": parameters['nfft'],
        "lowfreq": parameters['low_freq'],
        "highfreq": parameters['high_freq'],
        "preemph": 0.97
    }


class DataProcessor(object):
    """Handles loading, partitioning, and preparing audio training data."""

    def __init__(self):
        # Multi-processors
        self._multi_processors = None
        self._manager = None
        self._queue = None
        self._steps = None
        self._max_steps = None

        # Mapping table for speakerid and path list.
        self._data_index = []

        self._target_samplerate = 0

    def __del__(self):
        self.stop()

    def start(self, data_batch_size, data_queue_number, data_processor_number, is_random_seed=True):
        """ Create multi-processors for processing data. """
        def _task(feature_parameters, data_index, is_random_seed):
            total_set_size = len(data_index)
            for batch_offset in range(0, total_set_size, data_batch_size):
                mini_batch = data_index[batch_offset:batch_offset +
                                        data_batch_size]
                if batch_offset + data_batch_size > total_set_size:
                    mini_batch = mini_batch + \
                        [mini_batch[0]] * \
                        (batch_offset+data_batch_size-total_set_size)
                # try:
                # Pull the audio samples we'll use for training.
                features = AudioProcessor.get_data(
                    feature_parameters,
                    mini_batch,
                    is_random_seed)

                self._queue.put(features)
                # except Exception as e:  # pylint:
                #     # traceback.print_exc()
                #     # setattr(e, '__traceback__', None)
                #     # break
                #     raise e

        try:
            # Create shared training data collection queue.
            set_size = self.size

            self._manager = multiprocessing.Manager()
            self._queue = self._manager.Queue(
                maxsize=data_queue_number)

            # Create multi-processors for processing data.
            set_part_size = math.ceil(
                set_size/data_processor_number)
            inner_total_steps = 0
            self._multi_processors = []
            # Allocate partial data to different processors
            for part_offset in range(0, set_size, set_part_size):
                part_size = min(set_part_size,
                                set_size - part_offset)
                inner_total_steps += math.ceil(part_size/data_batch_size)

                partition = self._data_index[part_offset: part_offset+part_size]

                processor = multiprocessing.Process(
                    target=_task, args=(self._feature_parameters, partition, is_random_seed))

                processor.start()
                self._multi_processors.append(processor)

            self._steps = 0
            self._max_steps = inner_total_steps
        except Exception as e:
            self.stop()
            raise e

    def is_running(self):
        # print(self._steps, self._max_steps)
        return self._steps < self._max_steps

    @property
    def total_steps(self):
        return self._max_steps

    def get(self):
        """ Pull the audio features.
        Returns:
        tuple, (data is empty, features, feature number)
        """
        if not self.is_running():
            return (True, None, 0)
        else:
            self._steps += 1
            return (False, self._queue.get(), self._queue.qsize())

    def stop(self):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.
        """
        # if self.is_running():
        #     self._stop_event.set()
        if self._multi_processors is None:
            return

        for processor in self._multi_processors:
            if processor.is_alive():
                processor.terminate()

        if self._manager:
            self._manager.shutdown()

        self._multi_processors = []
        self._queue = None

    @property
    def size(self):
        """Calculates the number of samples in the dataset partition.
        Returns:
          Number of samples in the partition.
        """
        return len(self._data_index)


class AudioProcessor(DataProcessor):
    """Handles loading, partitioning, and preparing audio training data."""

    def __init__(self,
                 data_path,
                 model_settings):
        super().__init__()

        # feature_parameters Constructor
        self._feature_parameters = model_settings
        self.update_data_index(data_path)

    def update_data_index(self, data_path):
        if isinstance(data_path, str):
            for uuid, audio in tqdm(utils.read_scp(data_path), total=utils.lines(data_path)):
                self._data_index.append((uuid, audio))
            self._feature_parameters['data_type'] = 'eval'
        else:
            audio_path = data_path['audio']
            lbl_path = data_path['label']
            weight_path = data_path['weight']
            self._feature_parameters['speed_perb'] = data_path.get(
                'speed_perb', False)
            self._feature_parameters['data_type'] = 'train'

            uuid_audio = {}
            for uuid, audio in tqdm(utils.read_scp(audio_path), total=utils.lines(audio_path)):
                uuid_audio[uuid] = audio

            uuid_label = {}
            for uuid, labels in tqdm(utils.read_labels(lbl_path), total=utils.lines(lbl_path)):
                uuid_label[uuid] = labels

            uuid_weight = {}
            for uuid, weight in tqdm(utils.read_weight(weight_path), total=utils.lines(weight_path)):
                uuid_weight[uuid] = weight

            for uuid, audio in tqdm(uuid_audio.items()):
                if uuid in uuid_label and uuid in uuid_weight:
                    self._data_index.append(
                        (audio, uuid_label[uuid], uuid_weight[uuid]))

    @staticmethod
    def get_data(feature_parameters, mini_batch, is_random_seed):
        """Gather samples from the data set, applying transformations as needed.

        Args:
        feature_parameters: feature_parameters handle.
        mini_batch: list, list of sample data.

        Returns:
        tuple, Features(fbank, labels, weights)
        """
        def extract_feature(audio):
            audio_seg = AudioSegment.from_file(audio)
            if feature_parameters.get('speed_perb', False):
                audio_seg.speed_perturbation(rng=random_state)
            return python_speech_features.logfbank(audio_seg.samples,
                                                   samplerate=feature_parameters['samplerate'],
                                                   nfilt=feature_parameters['nfilt'],
                                                   winlen=feature_parameters['winlen'],
                                                   winstep=feature_parameters['winstep'],
                                                   nfft=feature_parameters['nfft'],
                                                   lowfreq=feature_parameters['lowfreq'],
                                                   highfreq=feature_parameters['highfreq'],
                                                   preemph=feature_parameters['preemph'])

        def padding_feature(seq_len_list, feature_list):
            # Padding features
            max_seq_len = max(seq_len_list)
            padding_zero_features = np.zeros(
                (batch_size, max_seq_len, feature_parameters['nfilt']), np.float32)
            for i, features in enumerate(feature_list):
                padding_zero_features[i, :seq_len_list[i], :] = features
            return padding_zero_features

        random_state = None
        if is_random_seed:
            random_state = np.random.RandomState(RANDOM_SEED)

        batch_size = len(mini_batch)

        if feature_parameters['data_type'] == 'eval':
            feature_list = []
            seq_len_list = []
            uuid_list = []
            for i, info in enumerate(mini_batch):
                uuid, audio = info
                fbank = extract_feature(audio)
                seq_len = fbank.shape[0]
                uuid_list.append(uuid)
                feature_list.append(fbank)
                seq_len_list.append(seq_len)

            # Padding features
            padding_zero_features = padding_feature(seq_len_list, feature_list)

            return {
                'uuid': uuid_list,
                "feature": padding_zero_features,
                'seq_len': seq_len_list
            }
        elif feature_parameters['data_type'] == 'train':
            # Use the processing graph we created earlier to repeatedly to generate the
            # final output sample data we'll use in training.
            feature_list = []
            label_list = []
            weight_list = []
            seq_len_list = []
            label_len_list = []
            for i, info in enumerate(mini_batch):
                audio, labels, weight = info
                fbank = extract_feature(audio)
                seq_len = fbank.shape[0]

                feature_list.append(fbank)
                label_list.append(labels)
                weight_list.append(weight)
                seq_len_list.append(seq_len)
                label_len_list.append(len(labels))

            # Padding features
            padding_zero_features = padding_feature(seq_len_list, feature_list)

            # Sparse type labels
            sparse_label = convert_sparse_label(
                label_list, batch_size, label_len_list)

            return {
                "feature": padding_zero_features,
                "label": sparse_label,
                "weight": weight_list,
                "seq_len": seq_len_list,
            }
        else:
            raise Exception('data_type %s not support' %(feature_parameters['data_type']))
