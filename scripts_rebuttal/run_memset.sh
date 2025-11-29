#!/usr/bin/env bash
set -euo pipefail

data_list=(
    arrhythmia breastw cardio cardiotocography glass census 
    # ionosphere pima wbc wine thyroid optdigits pendigits fraud 
    # satellite "satimage-2" campaign mammography shuttle 
    # nslkdd 
)

train_ratio=(1.0)
hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemSet"
memory_ratio=1.0
latent_ratio=1.0
patience=20
sche_gamma=0.98

for data in "${data_list[@]}"; do
    exp_name="$model_type-use_pos"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --use_pos_embedding \
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
        --patience "$patience" \
        --exp_name "$exp_name" 
done