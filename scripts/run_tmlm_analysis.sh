#!/usr/bin/env bash
set -euo pipefail

data_list=(   
    # fraud
    # shuttle
    
    # campaign
    # shuttle # 1 minutes

    # thyroid 
    optdigits 
    # wbc
    # glass
    # pima
    # breastw    
    # wine
    # cardiotocography
    # cardio 
    # satellite 
    # pendigits
    # arrhythmia 
    # thyroid 
    # ionosphere 
    # mammography 
    # census
) 

model_type="TMLM"
exp_name="$model_type"

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