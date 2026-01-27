#!/usr/bin/env bash
set -euo pipefail

data_list=(    
    wine
    glass
    wbc
    pima
    breastw    
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
    
    # nslkdd
    # fraud
    # census
) 

model_type="TAEDACLv3"
exp_name="TAEDACLv3-260126-cw0.1-ap0.95"
for data in "${data_list[@]}"; do
    echo "Running $exp_name on $data."
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --plot_latent \
        --plot_attn \
        --plot_score_histogram \
        --plot_input_recon
        # --plot_latent_norm_histogram \
        # --plot_recon \
done