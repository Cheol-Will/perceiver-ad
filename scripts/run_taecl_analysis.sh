#!/usr/bin/env bash
set -euo pipefail

data_list=(   
    # pima
    # wbc
    # thyroid 
    optdigits 
    # glass
    # breastw    
    # wine
    # cardiotocography
    # cardio 
    # satellite 
    # pendigits
    # campaign
    # arrhythmia 
    # thyroid 
    # ionosphere 
    # mammography 
    # shuttle # 1 minutes
    # census
) 

model_type="TAECL"
contrastive_loss_weight=0.01
temperature=0.2
exp_name="$model_type-temp$temperature-contra$contrastive_loss_weight"

for data in "${data_list[@]}"; do
    echo "Running $exp_name on $data."
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --contrastive_loss_weight $contrastive_loss_weight \
        --temperature $temperature \
        --plot_attn \
        --plot_recon \
        --plot_latent
        # --plot_latent_norm_histogram
        # --plot_contra_histogram \
        # --plot_histogram \
        # --plot_input_recon
done