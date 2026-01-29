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
    optdigits 
    cardio 
    cardiotocography 
    thyroid 
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

model_type="TDACL"
dacl_alpha=0.95
for data in "${data_list[@]}"; do
    exp_name="$model_type-260128-ap$dacl_alpha"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --dacl_alpha $dacl_alpha
done