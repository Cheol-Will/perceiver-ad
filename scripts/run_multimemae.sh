#!/usr/bin/env bash
set -euo pipefail

data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere hepatitis pima wbc wine thyroid optdigits pendigits satellite campaign mammography) # from MCM
model_type="MultiMemAE"

# mlp -------------------------------
hidden_dim_list=(256 128 64)
learning_rate_list=(0.001 0.005 0.01 0.05)
num_adapters_list=(8 16 32)

for data in "${data_list[@]}"; do
    for hidden_dim in "${hidden_dim_list[@]}"; do
        for learning_rate in "${learning_rate_list[@]}"; do
            for num_adapters in "${num_adapters_list[@]}"; do
                echo "Running $model_type data=$data dim=$hidden_dim learning_rate=$learning_rate num_adapters=$num_adapters"
                exp_name="$model_type-d$hidden_dim-lr$learning_rate-ad$num_adapters"
                python main.py \
                    --dataname "$data" \
                    --model_type $model_type \
                    --exp_name "$exp_name"\
                    --hidden_dim "$hidden_dim" \
                    --learning_rate "$learning_rate"\
                    --num_adapters "$num_adapters"
            done                    
        done
    done
done

temperature=0.1 # future work if this works.
for data in "${data_list[@]}"; do
    for hidden_dim in "${hidden_dim_list[@]}"; do
        for learning_rate in "${learning_rate_list[@]}"; do
            for num_adapters in "${num_adapters_list[@]}"; do
                echo "Running $model_type data=$data dim=$hidden_dim learning_rate=$learning_rate num_adapters=$num_adapters"
                exp_name="$model_type-d$hidden_dim-lr$learning_rate-ad$num_adapters-t$temperature"
                python main.py \
                    --dataname "$data" \
                    --model_type $model_type \
                    --exp_name "$exp_name"\
                    --hidden_dim "$hidden_dim" \
                    --learning_rate "$learning_rate"\
                    --num_adapters "$num_adapters"\
                    --temperature "$temperature"
            done                    
        done
    done
done