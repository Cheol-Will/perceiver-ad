#!/usr/bin/env bash
set -euo pipefail

data_list=(
    # wine glass wbc ionosphere arrhythmia breastw pima cardio cardiotocography thyroid 
    # optdigits satellite "satimage-2" pendigits mammography campaign shuttle 

    # longer training
    # nslkdd  # done
    # fraud
    # census
    # fraud nslkdd census
) 

model_type="OELATTE"
hidden_dim=32
oe_lambda=0.1 # default 1.0
oe_shuffle_ratio=0.3 # default 0.3
for data in "${data_list[@]}"; do
    exp_name="$model_type-d$hidden_dim-oe_lam$oe_lambda-oe_rat$oe_shuffle_ratio"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --hidden_dim "$hidden_dim" \
        --oe_lambda "$oe_lambda" \
        --oe_shuffle_ratio "$oe_shuffle_ratio" 
done