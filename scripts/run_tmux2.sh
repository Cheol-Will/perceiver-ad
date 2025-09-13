#!/usr/bin/env bash
set -euo pipefail

data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign mammography "satimage-2" nslkdd fraud  shuttle census) # from MCM
depth=6
hidden_dim=64
learning_rate=0.001
entropy_loss_weight=0.001
model_type="MemPAE"

for data in "${data_list[@]}"; do
    echo "Running $model_type data=$data dim=$hidden_dim learning_rate=$learning_rate weight sharing"
    exp_name="$model_type-ws-use_ent_score-ent$entropy_loss_weight-L$depth-d$hidden_dim-lr$learning_rate"
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --is_weight_sharing \
        --use_entropy_loss_as_score \
        --entropy_loss_weight $entropy_loss_weight\
        --depth $depth \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --exp_name "$exp_name"
done