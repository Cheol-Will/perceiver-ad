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
    cardio 
    cardiotocography 
    thyroid 
    
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

model_type="TMLMMixup"
# mixup_alpha_list=(0.1)
mixup_alpha_list=(0.2)
mixup_prob_list=(0.1 0.3 0.5)
for data in "${data_list[@]}"; do
    for mixup_alpha in "${mixup_alpha_list[@]}"; do
        for mixup_prob in "${mixup_prob_list[@]}"; do
            exp_name="$model_type-alpha$mixup_alpha-prob$mixup_prob"
            echo "Running $exp_name on $data."
            python main.py \
                --dataname "$data" \
                --model_type $model_type \
                --exp_name "$exp_name" \
                --mixup_alpha "$mixup_alpha" \
                --mixup_prob "$mixup_prob"
        done
    done
done