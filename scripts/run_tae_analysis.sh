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
    arrhythmia 
    thyroid 
    ionosphere 
    mammography 
    shuttle # 1 minutes
    # census
) 

model_type="TAE"
exp_name="TAE-tuned"
for data in "${data_list[@]}"; do
    echo "Running $exp_name on $data."
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --plot_recon \
        --plot_histogram \
        --plot_latent \
        --plot_input_recon
done