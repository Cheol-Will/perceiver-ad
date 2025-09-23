#!/usr/bin/env bash
set -euo pipefail

# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign mammography) # from MCM
data_list=(
    # thyroid
    # optdigits
    # "satimage-2"
    # satellite
    campaign
    nslkdd
    # fraud
    # census
) 
model_list=(DRL MCM Disent) 

train_ratio_list=(0.2 0.4 0.6 0.8)
for data in "${data_list[@]}"; do
    for train_ratio in "${train_ratio_list[@]}"; do
        for model_type in "${model_list[@]}"; do
            python main.py \
                --dataname "$data" \
                --model_type $model_type \
                --train_ratio "$train_ratio"
        done
    done
done