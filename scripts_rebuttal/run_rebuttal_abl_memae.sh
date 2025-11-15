#!/usr/bin/env bash
set -euo pipefail

# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite "satimage-2" campaign mammography shuttle nslkdd fraud census) # from MCM
data_list=("satimage-2" shuttle nslkdd fraud census) # from MCM

train_ratio=(1.0)
hidden_dim=256
temperature=0.1
learning_rate=0.001
model_type="MemAE"
for data in "${data_list[@]}"; do
    exp_name="$model_type-d$hidden_dim-lr$learning_rate-t$temperature"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name"
done