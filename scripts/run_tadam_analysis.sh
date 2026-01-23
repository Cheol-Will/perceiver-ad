#!/usr/bin/env bash
set -euo pipefail

data_list=(    
    wbc
    glass
    pima
    breastw    
    wine
    cardiotocography
    optdigits 
    cardio 
    satellite 
    pendigits
    campaign
    thyroid 
    ionosphere 
    mammography 
    shuttle # 1 minutes
    arrhythmia 
    # census
) 

model_type="TADAM"
exp_name="TADAM-default"
for data in "${data_list[@]}"; do
    echo "Running $exp_name on $data."
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --plot_attn
        # --plot_knn_histogram \
        # --plot_score_histogram
        # --plot_recon \
        # --plot_latent \
        # --plot_latent_norm_histogram
        # --plot_input_recon
done