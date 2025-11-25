#!/usr/bin/env bash
set -euo pipefail

# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite "satimage-2" campaign mammography shuttle nslkdd fraud census) # from MCM
data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite "satimage-2" campaign mammography shuttle nslkdd) # from MCM
# data_list=(fraud census)

train_ratio=(1.0)
hidden_dim=64
learning_rate=0.001
model_type="MemPAE"
latent_ratio=1.0
memory_ratio=1.0
temperature=0.05

for data in "${data_list[@]}"; do
    exp_name="$model_type-ws-local+global-sqrt_F$latent_ratio-sqrt_N$memory_ratio-d64-lr0.001-t$temperature"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --use_mask_token \
        --use_pos_enc_as_query \
        --latent_ratio $latent_ratio \
        --memory_ratio $memory_ratio \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --exp_name "$exp_name"
done