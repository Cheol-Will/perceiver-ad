#!/usr/bin/env bash
set -euo pipefail

data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite "satimage-2"  ) # from MCM
# data_list=(campaign mammography shuttle nslkdd)
# data_list=(fraud census)

model_type="MemPAE"
latent_ratio=1.0
memory_ratio=1.0
hidden_dim=64
learning_rate=0.01
temperature=0.1

for data in "${data_list[@]}"; do
    exp_name="MemPAE-attn-mlp-ws-lr$learning_rate"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --use_mask_token \
        --use_pos_enc_as_query \
        --latent_ratio $latent_ratio \
        --memory_ratio $memory_ratio \
        --mlp_mixer_decoder \
        --is_weight_sharing \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --temperature "$temperature" \
        --exp_name "$exp_name"
done