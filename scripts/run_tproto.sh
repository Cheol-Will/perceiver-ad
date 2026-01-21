#!/usr/bin/env bash
set -euo pipefail

data_list=(    
    wine 
    glass 
    wbc 
    ionosphere 
    arrhythmia 
    breastw 
    pima  
    cardio cardiotocography thyroid 
    
    optdigits 
    satellite 
    "satimage-2" 
    pendigits

    mammography 
    campaign 
    shuttle 
    fraud 
    nslkdd 
    census
) 

model_type="TProto"
# hidden_dim_list=(16)
# hidden_dim_list=(32)
# hidden_dim_list=(64)
# hidden_dim_list=(128)
# learning_rate_list=(0.001)
# learning_rate_list=(0.001 0.01)
# learning_rate_list=(0.1)
num_prototypes=5
temperature=0.1
for data in "${data_list[@]}"; do
    exp_name="$model_type-tuned-p$num_prototypes-temp$temperature"
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --num_prototypes "$num_prototypes" \
        --temperature "$temperature"
done
# for data in "${data_list[@]}"; do
#     for hidden_dim in "${hidden_dim_list[@]}"; do
#         for learning_rate in "${learning_rate_list[@]}"; do
#             exp_name="$model_type-d$hidden_dim-lr$learning_rate"
#             echo "Running $exp_name on $data."
#             python main.py \
#                 --dataname "$data" \
#                 --model_type $model_type \
#                 --exp_name "$exp_name" \
#                 --hidden_dim "$hidden_dim" \
#                 --learning_rate "$learning_rate"
#         done
#     done
# done