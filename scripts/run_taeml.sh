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
    # satellite 
    # "satimage-2" 
    # pendigits
    
    # mammography 
    # campaign 
    # shuttle 

    # fraud 

    nslkdd 
    census
) 

model_type="TAEML"
dacl_beta=0.8
contra_loss_weight_list=(0.1)
for data in "${data_list[@]}"; do
    exp_name="$model_type-260129-bt$dacl_beta"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --dacl_beta $dacl_beta \
        --runs 5
done