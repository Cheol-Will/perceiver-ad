#!/usr/bin/env bash
set -euo pipefail
model_list=(KNN MCM DRL Disent) 
data_list=(cardio cardiotocography pima wbc wine campaign) # from MCM
depth=4
hidden_dim=64
learning_rate=0.001
temperature=0.1
contamination_ratio_list=(0.01 0.03 0.05)

for model_type in "${model_list[@]}"; do
    for data in "${data_list[@]}"; do
        for contamination_ratio in "${contamination_ratio_list[@]}"; do
            echo "$model_type on $data with contamination_ratio=$contamination_ratio"
            python main.py \
                --dataname "$data" \
                --model_type $model_type \
                --contamination_ratio "$contamination_ratio"
        done
    done
done
