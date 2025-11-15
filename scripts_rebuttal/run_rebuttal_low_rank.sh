#!/usr/bin/env bash
set -euo pipefail

# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite "satimage-2" campaign mammography shuttle nslkdd fraud census) # from MCM
data_list=(census) # from MCM

train_ratio=(1.0)
hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemPAE"
memory_ratio=1.0
# latent = F
for data in "${data_list[@]}"; do
    exp_name="$model_type-ws-local+global-F-sqrt_N$memory_ratio-d$hidden_dim-lr$learning_rate-t$temperature"
    echo "Running $exp_name on $data."
    python main.py \
    --dataname "$data" \
    --model_type $model_type \
    --is_weight_sharing \
    --use_mask_token \
    --use_pos_enc_as_query \
    --use_latent_F \
    --memory_ratio $memory_ratio \
    --hidden_dim 64 \
    --learning_rate "$learning_rate" \
    --temperature "$temperature" \
    --exp_name "$exp_name"
done