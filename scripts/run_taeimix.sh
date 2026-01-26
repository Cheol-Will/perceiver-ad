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

model_type="TAEIMIX"
# imix_loss_weight_list=(0.1 0.01)
imix_loss_weight_list=(0.1)
# epochs=200
mixup_alpha=0.9
for data in "${data_list[@]}"; do
    for imix_loss_weight in "${imix_loss_weight_list[@]}"; do
    exp_name="$model_type-260125-iw$imix_loss_weight-ap$mixup_alpha"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --imix_loss_weight "$imix_loss_weight" \
        --mixup_alpha $mixup_alpha
    done
done