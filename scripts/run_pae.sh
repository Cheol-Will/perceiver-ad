#!/usr/bin/env bash
set -euo pipefail
data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign mammography) # from MCM
# data_list=(fraud nslkdd)
# data_list=("satimage-2" shuttle census)

hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="PAE"
for data in "${data_list[@]}"; do
    exp_name="$model_type-ws-pos_query+token-mlp_dec-d$hidden_dim-lr$learning_rate"
    echo "$exp_name on $data"
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --is_weight_sharing \
        --use_pos_enc_as_query \
        --use_mask_token \
        --mlp_mixer_decoder\
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --exp_name "$exp_name"
done