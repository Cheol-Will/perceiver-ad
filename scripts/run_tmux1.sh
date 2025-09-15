#!/usr/bin/env bash
set -euo pipefail

data_list=(
    contamination_0.01_18_Ionosphere_42
    contamination_0.03_18_Ionosphere_42
    contamination_0.05_18_Ionosphere_42
    contamination_0.01_32_shuttle_42
    contamination_0.03_32_shuttle_42
    contamination_0.05_32_shuttle_42
    contamination_0.01_38_thyroid_42
    contamination_0.03_38_thyroid_42
    contamination_0.05_38_thyroid_42
    contamination_0.01_26_optdigits_42
    contamination_0.03_26_optdigits_42
    contamination_0.05_26_optdigits_42
    contamination_0.01_23_mammography_42
    contamination_0.03_23_mammography_42
    contamination_0.05_23_mammography_42
)

model_list=(MCM DRL Disent) 
for data in "${data_list[@]}"; do
    for model_type in "${model_list[@]}"; do
        echo "$model_type on $data";
        python main.py \
            --dataname "$data"\
            --model_type "$model_type"\
            --exp_name "$model_type"
    done
done

