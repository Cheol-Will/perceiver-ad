#!/usr/bin/env bash
set -euo pipefail
# data_list=(arrhythmia breastw glass cardio cardiotocography pima wbc wine campaign) # from MCM
# data_list=(ionosphere  pendigits shuttle sattelite thyroid) # from MCM
data_list=(satellite shuttle) # from MCM

depth=4
hidden_dim=64
learning_rate=0.001
temperature=0.1
contamination_ratio_list=(0.01 0.03 0.05)
model_list=(KNN MCM Disent) 

for data in "${data_list[@]}"; do
    for contamination_ratio in "${contamination_ratio_list[@]}"; do
        for model_type in "${model_list[@]}"; do
            echo "$model_type on $data with contamination_ratio=$contamination_ratio"
            python main.py \
                --dataname "$data" \
                --model_type $model_type \
                --contamination_ratio "$contamination_ratio"
        done
    done
done
