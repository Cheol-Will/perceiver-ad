#!/usr/bin/env bash
set -euo pipefail
data_list=(
    # wine 
    # shuttle
    # "satimage-2"
    # satellite
    # ionosphere

    ####

    # wbc
    # cardiotocography
    # pendigits 
    cardio
    # mammography 

    # ####

    # glass 
    # breastw 
    # pima 
    # thyroid 
) 

train_ratio=(1.0)
model="RetAug"
exp_name="RetAugv2"
for data in "${data_list[@]}"; do
    python main.py \
        --dataname "$data" \
        --model $model \
        --exp_name $exp_name  
done