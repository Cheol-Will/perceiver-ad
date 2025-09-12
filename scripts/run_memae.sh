#!/usr/bin/env bash
set -euo pipefail

data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere hepatitis pima wbc wine thyroid optdigits pendigits satellite campaign mammography) # from MCM
train_ratio_list=(1.0)

# mlp -------------------------------
hidden_dim_list=(256 128 64)
learning_rate_list=(0.001 0.005 0.01 0.05)
model_type="MemAE"
temperature=0.1

for data in "${data_list[@]}"; do
    for hidden_dim in "${hidden_dim_list[@]}"; do
        for learning_rate in "${learning_rate_list[@]}"; do
            echo "Running $model_type data=$data dim=$hidden_dim learning_rate=$learning_rate"
            exp_name="$model_type-l2-d$hidden_dim-lr$learning_rate-t$temperature"
            python main.py \
                --dataname "$data" \
                --model_type $model_type \
                --exp_name "$exp_name"\
                --hidden_dim "$hidden_dim" \
                --learning_rate "$learning_rate"\
                --temperature "$temperature"\
                --sim_type "l2"
        done
    done
done