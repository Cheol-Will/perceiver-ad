#!/usr/bin/env bash
set -euo pipefail

data_list=(
    # cluster_anomalies_23_mammography_42
    # cluster_anomalies_26_optdigits_42
    # cluster_anomalies_29_Pima_42
    # cluster_anomalies_18_Ionosphere_42
    # cluster_anomalies_30_satellite_42
    # cluster_anomalies_31_satimage-2_42
    # cluster_anomalies_38_thyroid_42
    # cluster_anomalies_4_breastw_42
    # cluster_anomalies_6_cardio_42
    # cluster_anomalies_7_Cardiotocography_42

    local_anomalies_23_mammography_42
    local_anomalies_26_optdigits_42
    local_anomalies_29_Pima_42
    local_anomalies_18_Ionosphere_42
    local_anomalies_30_satellite_42
    local_anomalies_31_satimage-2_42
    local_anomalies_38_thyroid_42
    local_anomalies_4_breastw_42
    local_anomalies_6_cardio_42
    local_anomalies_7_Cardiotocography_42

    global_anomalies_23_mammography_42
    global_anomalies_26_optdigits_42
    global_anomalies_29_Pima_42
    global_anomalies_18_Ionosphere_42
    global_anomalies_30_satellite_42
    global_anomalies_31_satimage-2_42
    global_anomalies_38_thyroid_42
    global_anomalies_4_breastw_42
    global_anomalies_6_cardio_42
    global_anomalies_7_Cardiotocography_42

    # dependency_anomalies_23_mammography_42
    # dependency_anomalies_26_optdigits_42
    # dependency_anomalies_29_Pima_42
    # dependency_anomalies_18_Ionosphere_42
    # dependency_anomalies_30_satellite_42
    # dependency_anomalies_31_satimage-2_42
    # dependency_anomalies_38_thyroid_42
    # dependency_anomalies_4_breastw_42
    # dependency_anomalies_6_cardio_42
    # dependency_anomalies_7_Cardiotocography_42


)

# data_list=(
#     wine
# )

model_type="MemPAE"
latent_ratio=1.0
memory_ratio=1.0
hidden_dim=64
learning_rate=0.001
temperature=0.1

for data in "${data_list[@]}"; do
    # 'MemPAE-ws-pos_query+token-mlp_dec_mixer-d64-lr0.001-t0.1',
    exp_name="$model_type-ws-local+global-sqrt_F$latent_ratio-sqrt_N$memory_ratio-mlp_dec_mixer-d$hidden_dim-lr$learning_rate-t$temperature"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --use_mask_token \
        --use_pos_enc_as_query \
        --latent_ratio $latent_ratio \
        --memory_ratio $memory_ratio \
        --mlp_mixer_decoder \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --temperature "$temperature" \
        --exp_name "$exp_name"
done