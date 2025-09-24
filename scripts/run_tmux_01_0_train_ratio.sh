#!/usr/bin/env bash
set -euo pipefail

declare -A data_memories=(
    ["cardio"]=16
    ["cardiotocography"]=16
    ["mammography"]=64
    ["pendigits"]=32
    ["pima"]=8
    ["wbc"]=8
    ["wine"]=4
)

data_list=(cardio cardiotocography mammography pendigits pima wbc wine)

hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemPAE"
train_ratio_list=(0.2 0.4 0.6 0.8)

for data in "${data_list[@]}"; do
    num_memories=${data_memories[$data]}
    
    for train_ratio in "${train_ratio_list[@]}"; do
        exp_name="$model_type-train_ratio-ws-pos_query+token-d$hidden_dim-lr$learning_rate-t$temperature"
        echo "Running $exp_name on $data with num_memories=$num_memories."
        
        python main.py \
            --dataname "$data" \
            --model_type $model_type \
            --is_weight_sharing \
            --use_pos_enc_as_query \
            --use_mask_token \
            --hidden_dim "$hidden_dim" \
            --learning_rate "$learning_rate" \
            --temperature "$temperature" \
            --exp_name "$exp_name" \
            --num_memories "$num_memories" \
            --train_ratio "$train_ratio"
    done
done