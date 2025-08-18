#!/bin/bash

data_list=(arrhythmia cardio campaign Ionosphere optdigits) # high dim
# data_list=(annthyroid breastw mammography Pima thyroid) # low dim
# data_list=(annthyroid breastw mammography Pima thyroid) # low dim

# model_type=MCM
model_list=("MCM" "DRL" )
train_ratio_list=(1.0 0.5 0.1)

for model_type in "${model_list[@]}"; do
    for data in "${data_list[@]}"; do
        for train_ratio in "${train_ratio_list[@]}"; do
            python main.py \
                --dataname "$data" \
                --model_type "$model_type" \
                --exp_name "$model_type" \
                --train_ratio "$train_ratio" 
        done
    done
done