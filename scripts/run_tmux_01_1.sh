#!/usr/bin/env bash
set -euo pipefail

data_list=("satimage-2" shuttle) # from MCM
hidden_dim=64
learning_rate=0.001
temperature=0.01
model_type="MemPAE"
train_ratio_list=(1.0)
for data in "${data_list[@]}"; do
    for train_ratio in "${train_ratio_list[@]}"; do
        exp_name="$model_type-ws-pos_query+token-d$hidden_dim-lr$learning_rate-t$temperature"
        echo "Running $exp_name on $data."
        python main.py \
            --dataname "$data" \
            --model_type $model_type \
            --is_weight_sharing \
            --use_pos_enc_as_query \
            --use_mask_token \
            --hidden_dim "$hidden_dim" \
            --learning_rate "$learning_rate" \
            --temperature "$temperature" \
            --exp_name "$exp_name" \
            --train_ratio "$train_ratio"
    done
done