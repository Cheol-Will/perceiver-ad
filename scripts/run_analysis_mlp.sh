#!/bin/bash

data_list=(
    wine
    glass
    wbc
    ionosphere
    breastw
    pima
    cardio
    cardiotocography
    thyroid
    optdigits
    "satimage-2"
    satellite
    pendigits
    mammography
    campaign
    shuttle
    nslkdd
    fraud
)

model_type='AutoEncoder'
for data in "${data_list[@]}"; do
    exp_name="AutoEncoder"
    echo "$exp_name on $data"
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --plot_grad_z_x
done