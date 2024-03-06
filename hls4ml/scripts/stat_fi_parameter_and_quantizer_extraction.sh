#!/bin/bash
PRETRAINED_MODEL=./resnet_v1_eembc_quantized_tiny2_fkeras/model_best.h5

python3 stat_fi.py \
        --config ./tiny2_pynq-z2-fkeras.yml \
        --pretrained_model $PRETRAINED_MODEL \