#!/usr/bin/env bash
set -euo pipefail

data_list=(    
    # wine
    # glass 
    # wbc 
    # ionosphere 
    # arrhythmia 
    # breastw 
    # pima  
    # optdigits 
    
    # cardio 
    # cardiotocography 
    # thyroid 
    # "satimage-2" 
    # satellite 
    # pendigits
    
    # mammography 
    # campaign 
    # shuttle 
    
    # fraud 

    nslkdd 
    # census
) 

model_type="RECLv2"
dacl_alpha=0.95
dacl_beta=0.8
contra_loss_weight=0.05
ep=200
for data in "${data_list[@]}"; do
    exp_name="$model_type-260202-cw$contra_loss_weight-ap$dacl_alpha-bt$dacl_beta-ep$ep"

    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --contra_loss_weight "$contra_loss_weight" \
        --dacl_alpha $dacl_alpha \
        --dacl_beta $dacl_beta \
        --epochs $ep \
        --runs 5
done