# CAT-Tensorflow: Crf-based Asr Toolkit with TensorFlow implement

An extension of [thu-spmi](https://github.com/thu-spmi) [CAT](https://github.com/thu-spmi/CAT) for Tensorflow.

## Introduction

This is a modified version of [thu-spmi/CAT](https://github.com/thu-spmi/CAT). I just using Tensorflow to implement CRF ASR acoustic model training pipeline. More contents follow the [thu-spmi/CAT](https://github.com/thu-spmi/CAT) repo.

## Requirements

- [Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit)/[CuDNN](https://developer.nvidia.com/cudnn)

- [gcc/g++ 5.5.0](https://gcc.gnu.org)

- [Python3](https://www.python.org/download/releases/3.0/)

- [TensorFlow](https://www.tensorflow.org)

- [OpenFst](http://www.openfst.org)

- [Kaldi](https://kaldi-asr.org)

## Installation

Because CTC-CRF operator is based on CUDA Toolkit, so you should setting CUDA environment. For details, you can follow this [link](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) or TensorFlow official [link](https://www.tensorflow.org/install/pip?hl=zh-cn).

1. Install CUDA Toolkit

- Follow this [link](https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Linux&target_arch=x86_64) to download and install CUDA Toolkit 10.1 for your Linux distribution.
- Installation instructions can be found [here](https://docs.nvidia.com/cuda/archive/10.1/cuda-installation-guide-linux/index.html)

- Install CUDNN
  - Go to https://developer.nvidia.com/rdp/cudnn-download
  - Create a user profile if needed and log in
  - Select [cuDNN v7.6.5 (Nov 5, 2019), for CUDA 10.1](https://developer.nvidia.com/rdp/cudnn-download#a-collapse765-101)
  - Download [cuDNN v7.6.5 Library for Linux](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/cudnn-10.1-linux-x64-v7.6.5.32.tgz)
  - Follow the instructions under Section 2.3.1 of the [CuDNN Installation Guide](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux) to install CuDNN.

2. Environment Setup

   Append the following lines to `~/.bashrc` or `~/.zshrc`.

   ```shell
   export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
   export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   ```

3. Install TensorFlow with Anaconda virtual environment

   Create a virtual environment is recommended. You can choose [Conda](https://www.tensorflow.org/install/pip#conda) or [venv](https://docs.python.org/3/library/venv.html). Here I use Conda as an example.

   ```shell
   # Install TensorFlow/cuda/nvcc first, reference is here:
   conda create --name tf pip python==3.7
   conda activate tf
   conda install tensorflow-gpu==1.15.0
   ```

4. Install CTC-CRF TensorFlow wrapper [warp-ctc-crf](https://github.com/TeaPoly/warp-ctc-crf)

   setting your `TENSORFLOW_SRC_PATH` and `OPENFST`.

   NOTE: This is an example, please don't copy to your terminal:

   ```shell
   # Create a symlink libtensorflow_framework.so.1 which references the original file  libtensorflow_framework.so
   ln -s /home/huanglk/anaconda3_202002/envs/tf_subcomp/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so.1 /home/huanglk/anaconda3_202002/envs/tf_subcomp/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so
   
   # export TENSORFLOW_SRC_PATH
   export TENSORFLOW_SRC_PATH=/home/huanglk/anaconda3_202002/envs/tf_subcomp/lib/python3.7/site-packages/tensorflow_core/
   
   # export OPENFST
   export OPENFST=/usr/local/
   ```

   - It will compile three modules with gcc/g++, include `GPUCTC`, `PATHWEIGHT` and `GPUDEN`.
   - It is worth mentioning that if the version of gcc/g++ >= 5.0.0 and less than 6.0.0 will be helpful for following pipeline.
   - Finally, `Makefile` will exetucate `python3 ./setup.py install` for CTC-CRF TensorFlow wrapper.

   Now, you can install CTC-Crf TensorFlow wrapper `warp-ctc-crf`.

   ```shell
   # Install warp_ctc_crf
   cd warp_ctc_crf
   make -j 32
   ```

5. Install `cat_tensorflow` Asr Toolkit

     I using Conv2d->BiLstm as default acoustic model, if you wanna using your owner acoustic model. I suggest you replace `pythons3 setup.py install` with `python3 setup.py develop`.
     
     
     ```shell
     # Install Asr Toolkit  
     cd cat_tensorflow
     python3 ./setup.py install
     ```

## Usage

### Data preparation

First of all, you need prepare three files include audio path, text number and weight, they all using Kaldi script-file format (`${UUID} ${context}`). You can refer to `egs/test/data`.

And then you need change the configure file. I using python file as parameter setting.

The configuration file (`egs/test/config.py`) consists of three parts, namely basic settings, data path configuration, and model configuration. This is an template below:

```python
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
```

When calling training and testing scripts, you can also set the number of processes and queues for data capture, according to the configuration of the CPU. You can set the visible GPU indexs for calculate parallel, according to the configuration of the GPU.

### Model training

Here is an training exmaple (`egs/test/train.sh`).

```shell
python3 -m cat_tensorflow.train \
  --data_processor_number 12 \
  --data_queue_number 32 \
  --parameter ./config.py \
  --train_dir /tmp/crf_exp \
  --batch_size 8 \
  --GPU_settings "2,3" \
```

### Model Testing

The purpose of the testing script is to save the results of the logistic result. It's easy to apply language model to test the WER (word error rate).

Here is an testing exmaple (`egs/test/eval.sh`).

```shell
python3 -m cat_tensorflow.eval \
  --data_processor_number 12 \
  --data_queue_number 32 \
  --batch_size 12 \
  --GPU_settings "0,1" \
  --start_checkpoint /tmp/crf_exp/model/model.ckpt-23.index \
  --audio_scp ./data/wav.escp \
  --save_logit_dir ./exp/logit/data \
  --logit_scp ./exp/logit/logit.scp \
```

All scripts file in the `egs/test` directory.

## Lisense
Copyright © 2020 huanglk

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## References

CAT. https://github.com/thu-spmi/CAT.

