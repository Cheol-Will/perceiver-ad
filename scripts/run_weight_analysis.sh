#!/bin/bash

data_list=(wine_quality california house)
# data_list=(california)
# data_list=(wine_quality)
model_type='MemPAE'
# model_type='PAE'
hidden_dim=256
num_heads=8
learning_rate=0.001
temperature=0.1
for data in "${data_list[@]}"; do
    exp_name="$model_type-ws-pos_query+token-d$hidden_dim-h$num_heads-lr$learning_rate-t$temperature"
    # exp_name="$model_type-ws-pos_query+token-d$hidden_dim-lr$learning_rate"
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --is_weight_sharing \
        --use_pos_enc_as_query \
        --use_mask_token \
        --exp_name "$exp_name"\
        --hidden_dim "$hidden_dim" \
        --num_heads "$num_heads" \
        --learning_rate "$learning_rate" \
        --temperature "$temperature" \
        --compare_regresssion_with_attn
done