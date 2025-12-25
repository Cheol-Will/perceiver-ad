#!/usr/bin/env bash
set -euo pipefail

data_list=(    
    wine glass wbc ionosphere arrhythmia breastw pima cardio cardiotocography thyroid 
    optdigits satellite "satimage-2" pendigits mammography campaign shuttle 
    fraud nslkdd census
    # longer training
    # nslkdd  # done
    # fraud
    # census
    # fraud nslkdd census
) 

model_type="OELATTE"
# hidden_dim_list=(64)
# hidden_dim_list=(32)
# hidden_dim_list=(16)
oe_lambda_list=(0.1 1.0) # default 1.0
oe_shuffle_ratio_list=(0.1 0.3 0.5) # default 0.3
oe_lambda_memory=0.01

for data in "${data_list[@]}"; do
    for hidden_dim in "${hidden_dim_list[@]}"; do
        for oe_lambda in "${oe_lambda_list[@]}"; do
            for oe_shuffle_ratio in "${oe_shuffle_ratio_list[@]}"; do
                exp_name="$model_type-d$hidden_dim-oe_lam$oe_lambda-oe_rat$oe_shuffle_ratio-oe_lam_mem$oe_lambda_memory"
                echo "Running $exp_name on $data."
                python main.py \
                    --dataname "$data" \
                    --model_type $model_type \
                    --exp_name "$exp_name" \
                    --hidden_dim "$hidden_dim" \
                    --oe_lambda "$oe_lambda" \
                    --oe_shuffle_ratio "$oe_shuffle_ratio" \
                    --oe_lambda_memory "$oe_lambda_memory"

            done
        done
    done
done
