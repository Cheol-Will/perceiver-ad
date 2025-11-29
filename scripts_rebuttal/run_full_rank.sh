#!/usr/bin/env bash
set -euo pipefail

data_list=(
    # arrhythmia breastw cardio cardiotocography glass 
    census 
    # ionosphere pima wbc wine thyroid optdigits pendigits fraud 
    # satellite "satimage-2" 
    # campaign mammography 
    # shuttle 
    # nslkdd 
)

train_ratio=(1.0)
hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemPAE"
memory_ratio=1.0
patience=20
sche_gamma=0.99
# sche_gamma=1.0
for data in "${data_list[@]}"; do
    exp_name="LATTE-Full_rank-no_dec-p$patience-v2-sche_gamma$sche_gamma"
    exp_name="LATTE-Full_rank-no_dec-d$hidden_dim-lr$learning_rate-g$sche_gamma-t$temperature-p$patience"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --use_pos_enc_as_query \
        --is_weight_sharing \
        --use_latent_F \
        --not_use_decoder \
        --use_num_memories_power_2 \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --sche_gamma "$sche_gamma" \
        --temperature "$temperature" \
        --patience "$patience" \
        --exp_name "$exp_name" 
done