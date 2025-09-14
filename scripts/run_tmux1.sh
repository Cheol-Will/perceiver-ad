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


depth=6
hidden_dim=64
learning_rate=0.001
model_type="PAE"

for data in "${data_list[@]}"; do
    exp_name="$model_type-ws-L$depth-d$hidden_dim-lr$learning_rate"
    echo "$exp_name on $data"
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --is_weight_sharing \
        --depth $depth \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --exp_name "$exp_name"
done