#!/usr/bin/env bash
set -euo pipefail

data_list=(
    # overfitting
    # wine glass wbc ionosphere arrhythmia breastw pima cardio 
    # cardiotocography thyroid 
    # optdigits satellite "satimage-2" pendigits
    # mammography campaign shuttle 
    
    # fraud 
    nslkdd 
    # census
    # longer training
    # nslkdd  # done
    # fraud
    # census
) 

model_type="MBT"
# hidden_dim_list=(256)
hidden_dim_list=(128)
hidden_dim_list=(64)
# hidden_dim_list=(32)
# hidden_dim_list=(32 64 128)
# oe_lambda_list=(1.0)
# oe_lambda_list=(0.5) 
# oe_lambda_list=(0.2) 

top_k_list=(5)
# top_k_list=(10)
# top_k_list=(16)
# top_k_list=(32)
# top_k_list=(0)

# temperature_list=(0.1 1.0) # default 0.3
temperature_list=(0.1) # default 0.3
epochs=30
learning_rate=0.01
# patience=5
# patience=10
patience=15

for data in "${data_list[@]}"; do
    for hidden_dim in "${hidden_dim_list[@]}"; do
        for top_k in "${top_k_list[@]}"; do
            for temperature in "${temperature_list[@]}"; do
                # exp_name="$model_type-d$hidden_dim-top_k$top_k-temp$temperature-lr$learning_rate"
                exp_name="$model_type-d$hidden_dim-top_k$top_k-temp$temperature"
                echo "Running $exp_name on $data."
                python main.py \
                    --dataname "$data" \
                    --model_type $model_type \
                    --exp_name "$exp_name" \
                    --hidden_dim "$hidden_dim" \
                    --top_k "$top_k" \
                    --temperature "$temperature"
                    # --learning_rate "$learning_rate"
                    # --epochs "$epochs"
                    # --patience "$patience"
            done
        done
    done
done