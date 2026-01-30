#!/usr/bin/env bash
set -euo pipefail

data_list=(   
    optdigits 
    pima
    wbc
    thyroid 
    glass
    breastw    
    wine
    cardiotocography
    cardio 
    satellite 
    pendigits
    campaign
    arrhythmia 
    thyroid 
    ionosphere 
    mammography 
    # shuttle # 1 minutes
    # census
) 

model_type="TAECL"
contrastive_loss_weight=0.1
temperature=0.1
# exp_name="$model_type-temp$temperature-contra$contrastive_loss_weight"
exp_name="$model_type-tuned"

for data in "${data_list[@]}"; do
    echo "Running $exp_name on $data."
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --contrastive_loss_weight $contrastive_loss_weight \
        --temperature $temperature \
        --plot_input_recon \
        --plot_score_histogram
        # --plot_attn \
        # --plot_recon \
        # --plot_latent
done