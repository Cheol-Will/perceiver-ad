#!/bin/bash

# data_list=(satimage-2) # from MCM
data_list=(pendigits optdigits arrhythmia breastw glass ionosphere pima wbc wine cardio cardiotocography thyroid) # from MCM

# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign mammography ) # from MCM
# data_list=("satimage-2") # from MCM
# data_list=(pendigits) # from MCM

# data_list=(cardio cardiotocography) # from MCM

# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign) # from MCM
# data_list=(mammography census fraud shuttle nslkdd) # from MCM
# data_list=(pendigits) # from MCM
# data_list=(campaign) # from MCM
# data_list=(shuttle) # from MCM
data_list=(pendigits optdigits arrhythmia breastw glass ionosphere pima wbc wine cardio cardiotocography thyroid) # from MCM
# data_list=(breastw cardio ionosphere pendigits "satimage-2" shuttle wbc)
# data_list=(pendigits) # from MCM
# data_list=(cardiotocography) # from MCM
data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign) # from MCM

# data_list=(pima) # from MCM
# data_list=(wine wbc) # from MCM
# data_list=(cardiotocography) # from MCM
# data_list=(
#     # dependency_anomalies_45_wine_42
#     # dependency_anomalies_14_glass_42
#     # dependency_anomalies_42_WBC_42
#     # dependency_anomalies_18_Ionosphere_42
#     # dependency_anomalies_4_breastw_42
#     # dependency_anomalies_29_Pima_42
#     # dependency_anomalies_6_cardio_42
#     # dependency_anomalies_7_Cardiotocography_42
#     # dependency_anomalies_38_thyroid_42
#     # dependency_anomalies_26_optdigits_42
#     # dependency_anomalies_31_satimage-2_42
#     # dependency_anomalies_30_satellite_42
#     # dependency_anomalies_23_mammography_42
#     # dependency_anomalies_5_campaign_42
#     # dependency_anomalies_32_shuttle_42
#     mnist
# )



hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemPAE"
# latent_ratio=4.0
train_ratio_list=(1.0)
lambda=0.01
top_k=5
depth=5
for data in "${data_list[@]}"; do
    for train_ratio in "${train_ratio_list[@]}"; do
        exp_name="$model_type-ws-pos_query+token-np-lambda-top$top_k-L$depth-d$hidden_dim-lr$learning_rate-t$temperature"
        echo "Running $exp_name on $data."
        python analysis.py \
            --dataname "$data" \
            --model_type $model_type \
            --is_weight_sharing \
            --use_pos_enc_as_query \
            --use_mask_token \
            --not_use_power_of_two\
            --latent_loss_weight "$lambda"\
            --top_k $top_k\
            --depth $depth\
            --hidden_dim "$hidden_dim" \
            --learning_rate "$learning_rate" \
            --temperature "$temperature" \
            --exp_name "$exp_name" \
            --train_ratio "$train_ratio" \
            --plot_2x4
            # --get_single_samples
    done
done


# model_type='MemPAE'
# hidden_dim=64
# learning_rate=0.001
# temperature=0.1
# entropy_loss_weight=10
# # entropy_loss_weight=0.001
# for data in "${data_list[@]}"; do
#     # exp_name="$model_type-ws-pos_query+token-use_ent_score-ent$entropy_loss_weight-d$hidden_dim-lr$learning_rate-t$temperature"
#     exp_name="$model_type-ws-pos_query+token-d$hidden_dim-lr$learning_rate-t$temperature"
#     echo "$exp_name on $data"
#     python analysis.py \
#         --dataname "$data" \
#         --model_type $model_type \
#         --is_weight_sharing \
#         --use_pos_enc_as_query \
#         --use_mask_token \
#         --hidden_dim "$hidden_dim" \
#         --learning_rate "$learning_rate" \
#         --temperature "$temperature" \
#         --exp_name "$exp_name" \
#         --plot_2x4
#         # --plot_feature_reconstruction_distribution\
#         # --plot_histogram
#         # --plot_attn_pair
#         # --plot_attn_dec_memory \
#         # --plot_attn_single \
#         # --plot_attn_simple
#         # --use_entropy_loss_as_score \
#         # --entropy_loss_weight "$entropy_loss_weight"\

#         # --get_single_samples
#         # --plot_attn_simple \
#         # --plot_attn_all_heads \
#         # --plot_attn_all_depths \
#         # --plot_attn_everything \
#         # --get_single_samples
#         # --plot_attn_single
#         # --plot_hist_diff_memory_addressing
#         # --plot_tsne_latent_vs_memory
#         # --plot_attn_single
#         # --plot_tsne_latent_vs_memory
#         # --plot_tsne_single_class_with_memory
#         # --plot_umap_latent_vs_memory
#         # --use_latents_avg
#         # --plot_tsne_original_with_memory
#         # --plot_tsne_latent_vs_memory
#         # --plot_tsne_memory_separate \
#         # --plot_pos_encoding
#         # --plot_histogram
#         # --plot_recon
#         # --plot_tsne_recon
# done

#         # --use_entropy_loss_as_score \
#         # --entropy_loss_weight "$entropy_loss_weight"\
