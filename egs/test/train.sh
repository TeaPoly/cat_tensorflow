#!/bin/bash

python3 -m cat_tensorflow.train \
  --data_processor_number 12 \
  --data_queue_number 32 \
  --parameter ./config.py \
  --train_dir /tmp/crf_exp \
  --batch_size 8 \
  --GPU_settings "2,3" \

  # --start_checkpoint /tmp/crf_exp/default_interrupt.ckpt-26 \

