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
model_type="TADAMv2"
exp_name="TADAMv2"
for data in "${data_list[@]}"; do
    echo "Running $exp_name on $data"
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name $exp_name
done
