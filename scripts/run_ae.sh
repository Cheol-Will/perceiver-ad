#!/bin/bash

# Dataset from MCM and Disent-AD
# census fraud nslkdd shuttle 
# data_list=(arrhythmia optdigits breastw cardio campaign cardiotocography glass ionosphere mammography pendigits pima satellite "satimage-2" thyroid wbc wine) 
data_list=(census fraud nslkdd shuttle "satimage-2") 
model_list=(AutoEncoder) #
train_ratio_list=(1.0)

for data in "${data_list[@]}"; do
    for model_type in "${model_list[@]}"; do
        for train_ratio in "${train_ratio_list[@]}"; do
            python main.py \
                --dataname "$data" \
                --model_type "$model_type" \
                --exp_name "CustomAutoEncoder" \
                --train_ratio "$train_ratio"
        done
    done
done