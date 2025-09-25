#!/bin/bash
data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign mammography) # from MCM
data_list=(
    # dependency_anomalies_29_Pima_42
    # local_anomalies_29_Pima_42
    # global_anomalies_29_Pima_42
    # cluster_anomalies_29_Pima_42

    # dependency_anomalies_7_Cardiotocography_42
    # local_anomalies_7_Cardiotocography_42
    # global_anomalies_7_Cardiotocography_42
    # cluster_anomalies_7_Cardiotocography_42

    dependency_anomalies_18_Ionosphere_42
    local_anomalies_18_Ionosphere_42
    cluster_anomalies_18_Ionosphere_42
    global_anomalies_18_Ionosphere_42


    # dependency_anomalies_30_satellite_42
    # local_anomalies_30_satellite_42
    # cluster_anomalies_30_satellite_42
    # global_anomalies_30_satellite_42



    # dependency_anomalies_4_breastw_42
    # local_anomalies_4_breastw_42
    # cluster_anomalies_4_breastw_42
    # global_anomalies_4_breastw_42


    # dependency_anomalies_45_wine_42
    # dependency_anomalies_14_glass_42
    # dependency_anomalies_42_WBC_42
    # dependency_anomalies_18_Ionosphere_42
    # dependency_anomalies_4_breastw_42
    # dependency_anomalies_29_Pima_42
    # dependency_anomalies_6_cardio_42plot_tsne_input_vs_reconstruction
    # dependency_anomalies_7_Cardiotocography_42
    # dependency_anomalies_38_thyroid_42
    # dependency_anomalies_26_optdigits_42
    # dependency_anomalies_31_satimage-2_42
    # dependency_anomalies_30_satellite_42
    # dependency_anomalies_23_mammography_42
    # dependency_anomalies_5_campaign_42
    # dependency_anomalies_32_shuttle_42

    # global_anomalies_45_wine_42
    # global_anomalies_14_glass_42
    # global_anomalies_42_WBC_42
    # global_anomalies_18_Ionosphere_42
    # global_anomalies_4_breastw_42
    # global_anomalies_29_Pima_42
    # global_anomalies_6_cardio_42
    # global_anomalies_7_Cardiotocography_42
    # global_anomalies_38_thyroid_42
    # global_anomalies_26_optdigits_42
    # global_anomalies_31_satimage-2_42
    # global_anomalies_30_satellite_42
    # global_anomalies_23_mammography_42
    # global_anomalies_5_campaign_42
    # global_anomalies_32_shuttle_42


# ######################################
#     cluster_anomalies_45_wine_42
#     cluster_anomalies_14_glass_42
#     cluster_anomalies_42_WBC_42
#     cluster_anomalies_18_Ionosphere_42
#     cluster_anomalies_4_breastw_42
#     cluster_anomalies_29_Pima_42
#     cluster_anomalies_6_cardio_42
#     cluster_anomalies_7_Cardiotocography_42
#     cluster_anomalies_38_thyroid_42
#     cluster_anomalies_26_optdigits_42
#     cluster_anomalies_31_satimage-2_42
#     cluster_anomalies_30_satellite_42
#     cluster_anomalies_23_mammography_42
#     cluster_anomalies_5_campaign_42
#     cluster_anomalies_32_shuttle_42

#     local_anomalies_45_wine_42
#     local_anomalies_14_glass_42
#     local_anomalies_42_WBC_42
#     local_anomalies_18_Ionosphere_42
#     local_anomalies_4_breastw_42
#     local_anomalies_29_Pima_42
#     local_anomalies_6_cardio_42
#     local_anomalies_7_Cardiotocography_42
#     local_anomalies_38_thyroid_42
#     local_anomalies_26_optdigits_42
#     local_anomalies_31_satimage-2_42
#     local_anomalies_30_satellite_42
#     local_anomalies_23_mammography_42
#     local_anomalies_5_campaign_42
#     local_anomalies_32_shuttle_42

)


# data_list=(
#     # breastw # best SHAP vs Attn
#     # glass
#     # wine
#     # wbc
#     # "satimage-2"
#     # thyroid 
#     # ionosphere
#     # pendigits
#     # cardio
#     # mammography
#     # pima
#     # cardiotocography
#     ##
#     arrhythmia
#     optdigits 
#     satellite 
#     campaign 
#     shuttle
#     "satimage-2" 
# )


model_type='MemPAE'
hidden_dim=64
learning_rate=0.001
temperature=0.1
for data in "${data_list[@]}"; do
    exp_name="$model_type-ws-pos_query+token-d$hidden_dim-lr$learning_rate-t$temperature"
    echo "$exp_name on $data"
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --is_weight_sharing \
        --use_pos_enc_as_query \
        --use_mask_token \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --temperature "$temperature" \
        --exp_name "$exp_name" \
        --plot_tsne_input_vs_reconstruction
        # --plot_tsne_4types
        # --compare_shap_vs_anomaly_gradient_per_sample
        # --plot_tsne_memory_addressing
        # --compare_shap_vs_encoder_attention_per_sample
        # --visualize_attention_vs_shap
done