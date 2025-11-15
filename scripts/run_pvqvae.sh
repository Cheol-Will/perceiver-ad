#!/usr/bin/env bash
set -euo pipefail

data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite "satimage-2" campaign mammography shuttle nslkdd fraud census) # from MCM

train_ratio_list=(1.0)
hidden_dim_list=(16 32 64)
beta=1.0
vq_loss_weight=1.0
learning_rate_list=(0.001)
model_type="PVQVAE"

for data in "${data_list[@]}"; do
    for hidden_dim in "${hidden_dim_list[@]}"; do
        for learning_rate in "${learning_rate_list[@]}"; do
            exp_name="$model_type-ws-local+global-NOT_use_vq_loss$vq_loss_weight-sqrt_F-sqrt_NF-d$hidden_dim-lr$learning_rate-beta$beta"
            echo "Running $exp_name on $data."
            python main.py \
                --dataname "$data" \
                --model_type $model_type \
                --is_weight_sharing \
                --use_pos_enc_as_query \
                --use_mask_token \
                --vq_loss_weight "$vq_loss_weight" \
                --hidden_dim "$hidden_dim" \
                --learning_rate "$learning_rate"\
                --beta "$beta"\
                --exp_name "$exp_name"
        done
    done
done