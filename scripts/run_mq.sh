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

model_type="MQ"
hidden_dim_list=(16)
queue_size_list=(1024)
momentum_list=(0.999)
commitment_cost_list=(0.25)

for data in "${data_list[@]}"; do
    for hidden_dim in "${hidden_dim_list[@]}"; do
        for queue_size in "${queue_size_list[@]}"; do
            for momentum in "${momentum_list[@]}"; do
                for commitment_cost in "${commitment_cost_list[@]}"; do
                    exp_name="$model_type-d$hidden_dim-qs$queue_size-mo$momentum-com$commitment_cost"
                    echo "Running $exp_name on $data."
                    python main.py \
                        --dataname "$data" \
                        --model_type $model_type \
                        --exp_name "$exp_name" \
                        --hidden_dim "$hidden_dim" \
                        --queue_size "$queue_size" \
                        --momentum "$momentum" \
                        --commitment_cost "$commitment_cost"
                done
            done
        done
    done
done
