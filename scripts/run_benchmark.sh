#!/bin/bash

# Dataset from MCM and Disent-AD
# model_list=(IForest LOF OCSVM ECOD KNN PCA AutoEncoder DeepSVDD GOAD NeuTraL ICL MCM DRL Disent) 
# data_list=(arrhythmia optdigits breastw cardio campaign cardiotocography census fraud glass ionosphere mammography nslkdd  pendigits pima satellite "satimage-2" shuttle thyroid wbc wine) 
# data_list=(census fraud "satimage-2" shuttle) 

data_list=(arrhythmia cardio campaign cardiotocography mammography  pendigits pima  "satimage-2") 
model_list=(KNN Disent MCM DRL) 
train_ratio_list=(0.8)

for data in "${data_list[@]}"; do
    for model_type in "${model_list[@]}"; do
        for train_ratio in "${train_ratio_list[@]}"; do
            python main.py \
                --dataname "$data" \
                --model_type "$model_type" \
                --exp_name "$model_type" \
                --train_ratio "$train_ratio"
        done
    done
done