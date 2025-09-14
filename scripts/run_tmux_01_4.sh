#!/usr/bin/env bash
set -euo pipefail

# MemAE-d64-lr0.005-t0.1

data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign mammography "satimage-2" nslkdd fraud  shuttle census) # from MCM
hidden_dim=64
learning_rate=0.005
temperature=0.1
model_type="MemAE"

for data in "${data_list[@]}"; do
    exp_name="$model_type-d$hidden_dim-lr$learning_rate-t$temperature"
    echo "Run $exp_name on $data"
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --temperature "$temperature" \
        --exp_name "$exp_name"
done