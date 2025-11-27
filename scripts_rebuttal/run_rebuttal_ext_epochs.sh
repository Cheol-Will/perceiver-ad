#!/usr/bin/env bash
set -euo pipefail

data_list=(
    # arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid shuttle satellite mammography "satimage-2" optdigits pendigits 
    campaign 
    # nslkdd
    # census 
    # fraud  
) # from MCM
model_type="MemPAE"
latent_ratio=1.0
memory_ratio=1.0
hidden_dim=64
learning_rate=0.001
temperature=0.1
for data in "${data_list[@]}"; do
    exp_name="LATTE-Extended"
    echo "Running $exp_name on $data."
    python main_epoch.py \
        --dataname "$data" \
        --model_type $model_type \
        --use_mask_token \
        --use_pos_enc_as_query \
        --is_weight_sharing \
        --latent_ratio $latent_ratio \
        --memory_ratio $memory_ratio \
        --use_num_latents_power_2 \
        --use_num_memories_power_2 \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --temperature "$temperature" \
        --exp_name "$exp_name" 
        # --patience 5 
done