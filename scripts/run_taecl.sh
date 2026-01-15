#!/usr/bin/env bash
set -euo pipefail

data_list=(    
    wine 
    glass 
    wbc 
    ionosphere 
    arrhythmia 
    breastw 
    pima  
    cardio cardiotocography thyroid 
    
    optdigits 
    satellite 
    "satimage-2" 
    pendigits

    mammography 
    campaign 
    shuttle 
    fraud 
    nslkdd 
    census
) 

model_type="TAECL"
temperature_list=(0.2)
# temperature_list=(0.1)
# temperature_list=(1.0)
contrastive_loss_weight_list=(0.001)
# contrastive_loss_weight_list=(0.01)
# contrastive_loss_weight_list=(0.1)

for data in "${data_list[@]}"; do
    for temperature in "${temperature_list[@]}"; do
        for contrastive_loss_weight in "${contrastive_loss_weight_list[@]}"; do
            exp_name="$model_type-temp$temperature-contra$contrastive_loss_weight"
            echo "Running $exp_name on $data."
            python main.py \
                --dataname "$data" \
                --model_type $model_type \
                --exp_name "$exp_name" \
                --temperature "$temperature" \
                --contrastive_loss_weight "$contrastive_loss_weight"
        done
    done
done