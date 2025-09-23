#!/usr/bin/env bash
set -euo pipefail

data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite mammography "satimage-2" campaign census fraud nslkdd shuttle) # from MCM
# data_list=(census fraud )
# data_list=(nslkdd shuttle)
hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemPAE"
train_ratio_list=(1.0)
for data in "${data_list[@]}"; do
    for train_ratio in "${train_ratio_list[@]}"; do
        exp_name="$model_type-ws-pos_query+token-mlp_enc-d$hidden_dim-lr$learning_rate-t$temperature"
        echo "Running $exp_name on $data."
        python main.py \
            --dataname "$data" \
            --model_type $model_type \
            --is_weight_sharing \
            --use_pos_enc_as_query \
            --use_mask_token \
            --mlp_encoder \
            --hidden_dim "$hidden_dim" \
            --learning_rate "$learning_rate" \
            --exp_name "$exp_name" \
            --train_ratio "$train_ratio"
    done
done
