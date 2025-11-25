#!/usr/bin/env bash
set -euo pipefail

data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite "satimage-2" campaign mammography shuttle nslkdd fraud census) # from MCM
model_type="MemPAE"
latent_ratio=1.0
memory_ratio=1.0
hidden_dim=64
learning_rate=0.001
temperature=0.1
for data in "${data_list[@]}"; do
    exp_name="LATTE"
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
        --plot_first_latent_umap
        # --plot_first_latent_umap
        # --plot_first_latent_tsne
        # --plot_tsne_memory_addressing
done