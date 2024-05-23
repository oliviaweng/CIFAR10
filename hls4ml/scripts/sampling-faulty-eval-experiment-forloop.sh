#!/bin/bash
CONFIG=./tiny_pynq-z2_fkeras.yml
PRETRAINED_MODEL=./resnet_v1_eembc_quantized_tiny_fkeras/model_best.h5
CORRECT_IDX_FILE=./resnet_v1_eembc_quantized_tiny_fkeras/non_faulty_correct_indices.npy

# Sanity check
#bits=10600
bits=318208
VMs=318208
system=0
lbi=$((bits/VMs * system))
hbi=$((bits/VMs * system + bits/VMs))
for (( i=$lbi; i<$hbi ; i++ )); do 
echo "Sanity check ber = 0"
CUDA_VISIBLE_DEVICES="" python3 sampling_faulty_eval_experiment.py \
        --config $CONFIG \
        --pretrained_model $PRETRAINED_MODEL \
        --efd_fp "./temp_efd_emd_hesstrace_v0-799999_b${lbi}-${hbi}-forloop.log" \
        --efr_fp "./temp_efr_emd_hesstrace_v0-799999_b${lbi}-${hbi}-forloop.log" \
        --efx_overwrite 0 \
        --use_custom_bfr 1 \
        --bfr_start $i \
        --bfr_end   $((i+1)) \
        --bfr_step  1 \
        --num_val_inputs 2048 \
        --thread_id "$1"
        # --correct_idx_file $CORRECT_IDX_FILE \
exit
done
