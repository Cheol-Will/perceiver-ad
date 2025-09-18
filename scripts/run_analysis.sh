#!/bin/bash

data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid) # from MCM
model_type='MemPAE'
# model_type='PAE'
hidden_dim=64
learning_rate=0.001
temperature=0.1
for data in "${data_list[@]}"; do
    exp_name="$model_type-ws-pos_query+token-d$hidden_dim-lr$learning_rate-t$temperature"
    # exp_name="$model_type-ws-pos_query+token-d$hidden_dim-lr$learning_rate"
    echo "$exp_name on $data"
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --is_weight_sharing \
        --use_pos_enc_as_query \
        --use_mask_token \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --temperature "$temperature" \
        --exp_name "$exp_name"\
        --????????
        # --compare_regresssion_with_attn
done