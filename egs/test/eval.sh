#!/bin/bash

python3 -m cat_tensorflow.eval \
  --data_processor_number 12 \
  --data_queue_number 32 \
  --batch_size 12 \
  --GPU_settings "0,1" \
  --start_checkpoint /tmp/crf_exp/model/model.ckpt-23.index \
  --audio_scp ./data/wav.escp \
  --save_logit_dir ./exp/logit/data \
  --logit_scp ./exp/logit/logit.scp \

