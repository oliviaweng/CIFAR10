#!/bin/bash
PRETRAINED_MODEL=./resnet_v1_eembc_quantized_tiny2_fkeras/model_best.h5

NUM_VAL_INPUTS=(32 64 128 256 512 1024 2048 4096 8192 10000)

for i in ${NUM_VAL_INPUTS[@]}; do
python3 sampling_faulty_eval_experiment.py \
        --config ./tiny2_pynq-z2-fkeras.yml \
        --pretrained_model $PRETRAINED_MODEL \
        --batch_size 32 \
        --num_val_inputs $i
done