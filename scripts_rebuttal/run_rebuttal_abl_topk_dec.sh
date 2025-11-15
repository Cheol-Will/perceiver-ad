#!/usr/bin/env bash
set -euo pipefail

data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite "satimage-2" campaign mammography shuttle nslkdd fraud census) # from MCM

train_ratio=(1.0)
hidden_dim=64
learning_rate=0.001
model_type="MemPAE"
latent_ratio=1.0
memory_ratio=1.0
config_file_name="MemPAE_topk"
for data in "${data_list[@]}"; do
    exp_name="$model_type-mlp_dec_mixer"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --use_mask_token \
        --use_pos_enc_as_query \
        --latent_ratio $latent_ratio \
        --memory_ratio $memory_ratio \
        --mlp_mixer_decoder \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --config_file_name "$config_file_name" \
        --exp_name "$exp_name"
done