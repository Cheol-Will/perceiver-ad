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
model_type="TAEDACL"
exp_name="TAEDACL-260125-bw0.1-ap0.9-ph"
pth_dir_name="TAEDACL-260125-bw0.1-ap0.9"
for data in "${data_list[@]}"; do
    echo "Running $exp_name on $data"
    python test.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name $exp_name \
        --pth_dir_name $pth_dir_name
done