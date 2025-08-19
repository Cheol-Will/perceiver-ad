#!/bin/bash

data_list=(arrhythmia cardio campaign Ionosphere optdigits) # high dim
# data_list=(annthyroid breastw mammography Pima thyroid) # low dim
# data_list=(annthyroid breastw mammography Pima thyroid) # low dim

# model_type=MCM
# model_list=("MCM" "DRL")
model_list=("MCM" "DRL")
lr_list=(0.1 0.05 0.025 0.01)
train_ratio_list=(1.0 0.5 0.1)

for model_type in "${model_list[@]}"; do
    for data in "${data_list[@]}"; do
        for train_ratio in "${train_ratio_list[@]}"; do
            for learning_rate in "${lr_list[@]}"; do
                python main.py \
                    --dataname "$data" \
                    --model_type "$model_type" \
                    --exp_name "$model_type" \
                    --train_ratio "$train_ratio" \
                    --learning_rate "$learning_rate" 
            done
        done
    done
done