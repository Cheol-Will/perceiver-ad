#!/usr/bin/env bash
set -euo pipefail

data_list=(
    ##################################
    local_anomalies_45_wine_42
    local_anomalies_14_glass_42
    local_anomalies_42_WBC_42
    local_anomalies_18_Ionosphere_42
    local_anomalies_4_breastw_42
    local_anomalies_29_Pima_42
    local_anomalies_6_cardio_42
    local_anomalies_7_Cardiotocography_42
    local_anomalies_38_thyroid_42
    local_anomalies_26_optdigits_42
    local_anomalies_31_satimage-2_42
    local_anomalies_30_satellite_42
    local_anomalies_23_mammography_42
    local_anomalies_5_campaign_42
    local_anomalies_32_shuttle_42
    # local_anomalies_13_fraud_42
    # local_anomalies_9_census_42
    ##################################

    ##################################
    dependency_anomalies_45_wine_42
    dependency_anomalies_14_glass_42
    dependency_anomalies_42_WBC_42
    dependency_anomalies_18_Ionosphere_42
    dependency_anomalies_4_breastw_42
    dependency_anomalies_29_Pima_42
    dependency_anomalies_6_cardio_42
    dependency_anomalies_7_Cardiotocography_42
    dependency_anomalies_38_thyroid_42
    dependency_anomalies_26_optdigits_42
    dependency_anomalies_31_satimage-2_42
    dependency_anomalies_30_satellite_42
    dependency_anomalies_23_mammography_42
    dependency_anomalies_5_campaign_42
    dependency_anomalies_32_shuttle_42
    # dependency_anomalies_13_fraud_42
    # dependency_anomalies_9_census_42
    ##################################
)

hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemPAE"
depth=6
for data in "${data_list[@]}"; do
    echo "Running $model_type data=$data dim=$hidden_dim learning_rate=$learning_rate weight sharing"
    exp_name="$model_type-ws-pos_query+token-np-L$depth-d$hidden_dim-lr$learning_rate-t$temperature"
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --is_weight_sharing \
        --use_pos_enc_as_query \
        --use_mask_token \
        --not_use_power_of_two \
        --depth "$depth" \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --temperature "$temperature" \
        --exp_name "$exp_name"
done