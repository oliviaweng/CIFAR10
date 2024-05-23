#!/bin/bash
CONFIG=./tiny_pynq-z2_fkeras.yml
PRETRAINED_MODEL=./resnet_v1_eembc_quantized_tiny_fkeras/model_best.h5
CORRECT_IDX_FILE=./resnet_v1_eembc_quantized_tiny_fkeras/non_faulty_correct_indices.npy
IEU_EFX_DIR="/home/anmeza/GitHub/FKeras-Experiments/data"
 
model_id=$1
vinputs=$2
vsystem=$3 
lbi=$4
hbi=$5
git_step=$6

for (( i=$lbi; i<$hbi ; i++ )); do 
echo "Sanity check ber = 0"
python3 fault_injection.py \
        --config $CONFIG \
        --pretrained_model $PRETRAINED_MODEL \
        --efd_fp "./efd_emd_hesstrace_v0-799999_b${lbi}-${hbi}-forloop.log" \
        --efr_fp "./efr_emd_hesstrace_v0-799999_b${lbi}-${hbi}-forloop.log" \
        --efx_overwrite 0 \
        --use_custom_bfr 1 \
        --bfr_start $i \
        --bfr_end   $((i+1)) \
        --bfr_step  1 \
        --num_val_inputs $vinputs \
        --correct_idx_file $CORRECT_IDX_FILE \
        --ieu_model_id $model_id \
        --ieu_vinputs $vinputs \
        --ieu_vsystem_id $vsystem \
        --ieu_efx_dir $IEU_EFX_DIR \
        --ieu_pefr_name "${model_id}_pefr_vinputs${vinputs}_vsystem${vsystem}_b${lbi}-${hbi}.pkl" \
        --ieu_pefd_name "${model_id}_pefd_vinputs${vinputs}_vsystem${vsystem}_b${lbi}-${hbi}.pkl" \
        --ieu_git_step $git_step \
        --ieu_lbi $lbi
done
