#!/usr/bin/env bash
set -euo pipefail

# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite "satimage-2" campaign mammography shuttle nslkdd fraud census) # from MCM
data_list=(
    local_anomalies_18_Ionosphere_42
    # breastw
    # ionosphere
    # pima 
    # wbc 
    # wine
)
model_type="MemPAE"
latent_ratio=1.0
memory_ratio=1.0
hidden_dim=64
learning_rate=0.001
temperature=0.1
for data in "${data_list[@]}"; do
    exp_name="$model_type-ws-local+global-sqrt_F$latent_ratio-sqrt_N$memory_ratio-d$hidden_dim-lr$learning_rate-t$temperature"
    echo "Running $exp_name on $data."
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --use_mask_token \
        --use_pos_enc_as_query \
        --latent_ratio $latent_ratio \
        --memory_ratio $memory_ratio \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --temperature "$temperature" \
        --exp_name "$exp_name" \
        --plot_umap_input_vs_reconstruction 
done
# data_list=(
#     dependency_anomalies_18_Ionosphere_42
# )
# for data in "${data_list[@]}"; do
#     exp_name="$model_type-ws-local+global-sqrt_F$latent_ratio-sqrt_N$memory_ratio-d$hidden_dim-lr$learning_rate-t$temperature"
#     echo "Running $exp_name on $data."
#     python analysis.py \
#         --dataname "$data" \
#         --model_type $model_type \
#         --use_mask_token \
#         --use_pos_enc_as_query \
#         --latent_ratio $latent_ratio \
#         --memory_ratio $memory_ratio \
#         --hidden_dim "$hidden_dim" \
#         --learning_rate "$learning_rate" \
#         --temperature "$temperature" \
#         --exp_name "$exp_name" \
#         --plot_umap_input_vs_reconstruction 
# done