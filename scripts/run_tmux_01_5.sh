#!/usr/bin/env bash
set -euo pipefail

#!/bin/bash
data_list=(
    ##################################
    global_anomalies_18_Ionosphere_42
    global_anomalies_38_thyroid_42
    global_anomalies_32_shuttle_42
    global_anomalies_26_optdigits_42
    global_anomalies_23_mammography_42
    cluster_anomalies_18_Ionosphere_42
    cluster_anomalies_38_thyroid_42
    cluster_anomalies_32_shuttle_42
    cluster_anomalies_26_optdigits_42
    cluster_anomalies_23_mammography_42
    dependency_anomalies_18_Ionosphere_42
    dependency_anomalies_38_thyroid_42
    dependency_anomalies_32_shuttle_42
    dependency_anomalies_26_optdigits_42
    dependency_anomalies_23_mammography_42
    local_anomalies_18_Ionosphere_42
    local_anomalies_38_thyroid_42
    local_anomalies_32_shuttle_42
    local_anomalies_26_optdigits_42
    local_anomalies_23_mammography_42
    ##################################
)


depth=4
hidden_dim=64
learning_rate=0.001
entropy_loss_weight=0.001
temperature=0.1
model_type="MemPAE"

for data in "${data_list[@]}"; do
    exp_name="$model_type-ws-pos_query+token-large_mem-L$depth-d$hidden_dim-lr$learning_rate-t$temperature"
    echo "$exp_name on $data"
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --is_weight_sharing \
        --use_pos_enc_as_query \
        --use_mask_token \
        --num_memories_not_use_power_of_two \
        --depth $depth \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --temperature "$temperature" \
        --exp_name "$exp_name"
done