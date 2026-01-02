#!/usr/bin/env bash
set -euo pipefail

data_list=(    
    # wine 
    # glass # 0.3057
    # wbc # 0.7887
    ionosphere # 0.9795
    
    # arrhythmia 
    # breastw 
    # pima  # too small-scale datset
    # cardio cardiotocography thyroid optdigits 
    # satellite "satimage-2" pendigits
    # mammography 
    # campaign 
    # shuttle 
    # fraud 
    # nslkdd 
    # census
) 

model_type="MQ"
hidden_dim_list=(32)
hidden_dim_list=(64)
hidden_dim_list=(128)
# hidden_dim_list=(256)
queue_size_list=(16384) # 2048 * 8
momentum=0.999
# top_k_list=(0)
top_k_list=(5)
# top_k_list=(10)
# top_k_list=(16)
# top_k_list=(32)
# top_k_list=(128)
# temperature_list=(0.1 1.0) 
temperature_list=(0.1) 
learning_rate=0.01

for data in "${data_list[@]}"; do
    for hidden_dim in "${hidden_dim_list[@]}"; do
        for queue_size in "${queue_size_list[@]}"; do
            for top_k in "${top_k_list[@]}"; do
                for temperature in "${temperature_list[@]}"; do
                    # exp_name="$model_type-d$hidden_dim-qs$queue_size-mo$momentum-top_k$top_k-temp$temperature-lr$learning_rate"
                    exp_name="$model_type-d$hidden_dim-qs$queue_size-mo$momentum-top_k$top_k-temp$temperature"
                    echo "Running $exp_name on $data."
                    python main.py \
                        --dataname "$data" \
                        --model_type $model_type \
                        --exp_name "$exp_name" \
                        --hidden_dim "$hidden_dim" \
                        --momentum "$momentum" \
                        --top_k "$top_k" \
                        --temperature "$temperature"
                        # --learning_rate "$learning_rate"
                        # --queue_size "$queue_size" \
                done
            done
        done
    done
done