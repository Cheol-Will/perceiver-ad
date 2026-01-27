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

model_type="TAEIMIXv2"
# imix_loss_weight_list=(0.1 0.01)
contra_loss_weight=(0.1)
# mixup_alpha=0.9
mixup_alpha=0.05
for data in "${data_list[@]}"; do
    for contra_loss_weight in "${contra_loss_weight[@]}"; do
    exp_name="$model_type-260126-cw$contra_loss_weight-ap$mixup_alpha"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --contra_loss_weight "$contra_loss_weight" \
        --mixup_alpha $mixup_alpha
    done
done