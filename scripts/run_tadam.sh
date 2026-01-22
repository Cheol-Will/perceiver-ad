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
    optdigits 

    cardio 
    cardiotocography 
    thyroid 
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
model_type="TADAM"
exp_name="TADAM-default-reproduce"
for data in "${data_list[@]}"; do
    echo "Running $exp_name on $data"
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name $exp_name
done

# hidden_dim_list=(128)
# lr_list=(0.01 0.001)
# model_type="TADAM"
# for data in "${data_list[@]}"; do
#     for hidden_dim in "${hidden_dim_list[@]}"; do
#         for lr in "${lr_list[@]}"; do
#         exp_name="TADAM-d$hidden_dim-lr$lr"
#         echo "Running $exp_name on $data."
#         python main.py \
#             --dataname "$data" \
#             --model_type $model_type \
#             --exp_name "$exp_name" \
#             --hidden_dim "$hidden_dim" \
#             --learning_rate "$lr"
#         done
#     done
# done

