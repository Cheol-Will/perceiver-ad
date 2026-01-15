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
    # cardio cardiotocography 
    thyroid 
    optdigits 
    satellite 
    "satimage-2" 
    pendigits

    # mammography 
    # campaign 
    # shuttle 
    # fraud 

    # nslkdd 
    # census
) 

model_type="TCL"
temperature_list=(0.1)
mixup_alpha_list=(0)
# mixup_alpha_list=(0.1)
# mixup_alpha_list=(1.0)

for data in "${data_list[@]}"; do
    for temperature in "${temperature_list[@]}"; do
        for mixup_alpha in "${mixup_alpha_list[@]}"; do
            exp_name="$model_type-temp$temperature-mixup_alpha$mixup_alpha"
            echo "Running $exp_name on $data."
            python main.py \
                --dataname "$data" \
                --model_type $model_type \
                --exp_name "$exp_name" \
                --temperature "$temperature" \
                --mixup_alpha "$mixup_alpha"
        done
    done
done