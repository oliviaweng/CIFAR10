#!/bin/bash
PRETRAINED_MODEL=./resnet_v1_eembc_quantized_tiny2_fkeras/model_best_untrained.h5

# Sanity check
#bits=10600
bits=460656
VMs=8
system=0
lbi=$((bits/VMs * system))
hbi=$((bits/VMs * system + bits/VMs))
for (( i=$lbi; i<$hbi ; i++ )); do 
echo "Sanity check ber = 0"
CUDA_VISIBLE_DEVICES="" python3 sampling_faulty_eval_experiment.py \
        --config ./tiny2_pynq-z2-fkeras.yml \
        --pretrained_model $PRETRAINED_MODEL \
        --efd_fp "./efd_emd_hesstrace_v0-799999_b${lbi}-${hbi}-forloop.log" \
        --efr_fp "./efr_emd_hesstrace_v0-799999_b${lbi}-${hbi}-forloop.log" \
        --efx_overwrite 0 \
        --use_custom_bfr 1 \
        --bfr_start $i \
        --bfr_end   $((i+1)) \
        --bfr_step  1 \
        --num_val_inputs 32
exit
done