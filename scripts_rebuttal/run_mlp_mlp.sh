#!/usr/bin/env bash
set -euo pipefail

# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite "satimage-2" campaign mammography shuttle nslkdd) # from MCM
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

    # local_anomalies_23_mammography_42
    # local_anomalies_26_optdigits_42
    # local_anomalies_29_Pima_42
    # local_anomalies_18_Ionosphere_42
    # local_anomalies_30_satellite_42
    # local_anomalies_31_satimage-2_42
    # local_anomalies_38_thyroid_42
    # local_anomalies_4_breastw_42
    # local_anomalies_6_cardio_42
    # local_anomalies_7_Cardiotocography_42

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
)
model_type="MemPAE"
latent_ratio=1.0
memory_ratio=1.0
hidden_dim=256
learning_rate=0.1
temperature=0.1

for data in "${data_list[@]}"; do
    # exp_name="MemPAE-attn-mlp-no_mem-v2"
    # exp_name="MemPAE-mlp-mlp-v3-t0.2"
    # exp_name="MemPAE-mlp-mlp-v3-no_mem"
    # exp_name="MemPAE-mlp-mlp-v3"
    exp_name="MemPAE-mlp-mlp-d$hidden_dim-lr$learning_rate"
    # exp_name="MemAE-d$hidden_dim-lr#$learning_rate"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --use_mask_token \
        --use_pos_enc_as_query \
        --latent_ratio $latent_ratio \
        --memory_ratio $memory_ratio \
        --mlp_mixer_encoder \
        --mlp_mixer_decoder \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --temperature "$temperature" \
        --exp_name "$exp_name"
done