#!/usr/bin/env bash
set -euo pipefail

# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere hepatitis pima wbc wine thyroid optdigits pendigits satellite campaign mammography) # from MCM

data_list=(fraud  shuttle census) # from MCM

hidden_dim=64
learning_rate=0.005
temperature=0.1
model_type="MCMPAE"
for data in "${data_list[@]}"; do
    exp_name="$model_type-ws-pos_query+token-d$hidden_dim-lr$learning_rate"
    echo "$exp_name on $data"
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --is_weight_sharing \
        --use_pos_enc_as_query \
        --use_mask_token \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --exp_name "$exp_name"
done