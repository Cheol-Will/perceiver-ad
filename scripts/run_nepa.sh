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
    satellite "satimage-2" pendigits

    mammography 
    campaign 
    shuttle 
    fraud 
    nslkdd 
    census
) 

model_type="NEPA"
hidden_dim_list=(16)
learning_rate=0.01

for data in "${data_list[@]}"; do
    for hidden_dim in "${hidden_dim_list[@]}"; do
        exp_name="$model_type-d$hidden_dim"
        echo "Running $exp_name on $data."
        python main.py \
            --dataname "$data" \
            --model_type $model_type \
            --exp_name "$exp_name" \
            --hidden_dim "$hidden_dim" 
    done
done