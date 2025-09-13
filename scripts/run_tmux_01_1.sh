#!/usr/bin/env bash
set -euo pipefail

data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign mammography "satimage-2" nslkdd fraud  shuttle census) # from MCM
top_k=1
depth=4
hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemPAE"

for data in "${data_list[@]}"; do
    echo "Running $model_type data=$data dim=$hidden_dim learning_rate=$learning_rate weight sharing"
    exp_name="$model_type-ws-pos_query+token-top$top_k-L$depth-d$hidden_dim-lr$learning_rate-t$temperature"
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --is_weight_sharing \
        --use_pos_enc_as_query \
        --use_mask_token \
        --top_k "$top_k" \
        --depth "$depth" \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --temperature "$temperature" \
        --exp_name "$exp_name"
done