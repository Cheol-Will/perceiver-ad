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
    # nslkdd 
    census
) 

model_type="TAEDACLv3"
dacl_alpha=0.95
contra_loss_weight_list=(0.1)
# contra_loss_weight_list=(0.01)
for data in "${data_list[@]}"; do
    for contra_loss_weight in "${contra_loss_weight_list[@]}"; do
        exp_name="$model_type-260126-cw$contra_loss_weight-ap$dacl_alpha"
        echo "Running $exp_name on $data."
        python main.py \
            --dataname "$data" \
            --model_type $model_type \
            --exp_name "$exp_name" \
            --contra_loss_weight "$contra_loss_weight" \
            --dacl_alpha $dacl_alpha
    done
done