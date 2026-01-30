#!/usr/bin/env bash
set -euo pipefail

data_list=(    
    wine
    optdigits
    glass
    wbc
    pima
    breastw    
    cardiotocography
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

model_type="TAEDACLv4"
dacl_alpha=0.95
dacl_beta=0.8
contra_loss_weight=0.1
# exp_name="$model_type-260126-cw$contra_loss_weight-ap$dacl_alpha-bt$dacl_beta"
exp_name="$model_type-260130-cw$contra_loss_weight-ap$dacl_alpha-bt$dacl_beta-use_bn"

for data in "${data_list[@]}"; do
    echo "Running $exp_name on $data."
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --use_bn \
        --plot_input_recon \
        --plot_score_histogram
        # --plot_repeat_recon_historgram
        # --plot_latent \
        # --plot_attn \
done