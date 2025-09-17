#!/usr/bin/env bash
set -euo pipefail
data_list=(cardio cardiotocography pima wbc wine campaign "satimage-2") # from MCM

depth=4
hidden_dim=64
learning_rate=0.001
temperature=0.1
contamination_ratio_list=(0.01 0.03 0.05)
model_type="MemPAE"

for data in "${data_list[@]}"; do
    for contamination_ratio in "${contamination_ratio_list[@]}"; do
        exp_name="$model_type-ws-pos_query+token-d$hidden_dim-lr$learning_rate-t$temperature"
        echo "$exp_name on $data with contamination_ratio=$contamination_ratio"
        python main.py \
            --dataname "$data" \
            --model_type $model_type \
            --is_weight_sharing \
            --use_pos_enc_as_query \
            --use_mask_token \
            --hidden_dim "$hidden_dim" \
            --learning_rate "$learning_rate" \
            --temperature  "$temperature" \
            --contamination_ratio "$contamination_ratio" \
            --exp_name "$exp_name"
    done
done
