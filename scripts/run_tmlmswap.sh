#!/usr/bin/env bash
set -euo pipefail

data_list=(    

    # wine 
    # glass 
    # wbc 
    # ionosphere 
    # arrhythmia 
    # breastw 
    # pima  
    # optdigits 
    # cardio 
    # cardiotocography 
    # thyroid 
    
    # optdigits 
    # satellite 
    # "satimage-2" 
    # pendigits

    # mammography 
    # campaign 
    # shuttle 

    # fraud 

    # nslkdd 

    census
) 

model_type="TMLMSwap"
num_eval_repeat=50
swap_ratio_list=(0.1 0.3 0.5)
for data in "${data_list[@]}"; do
    for swap_ratio in "${swap_ratio_list[@]}"; do
        exp_name="$model_type-default-swap$swap_ratio-r$num_eval_repeat"
        echo "Running $exp_name on $data."
        python main.py \
            --dataname "$data" \
            --model_type $model_type \
            --exp_name "$exp_name" \
            --num_eval_repeat "$num_eval_repeat" \
            --swap_ratio "$swap_ratio"
    done
done


# model_type="TMLM"
# hidden_dim=64
# learning_rate_list=(0.001)
# # learning_rate_list=(0.01)
# # mask_ratio=0.1
# # mask_ratio=0.2
# # mask_ratio=0.3
# mask_ratio_list=(0.1 0.3 0.5)
# num_eval_repeat=50

# for data in "${data_list[@]}"; do
#     for mask_ratio in "${mask_ratio_list[@]}"; do
#         for learning_rate in "${learning_rate_list[@]}"; do
#             exp_name="$model_type-d$hidden_dim-lr$learning_rate-mask$mask_ratio-r$num_eval_repeat"
#             echo "Running $exp_name on $data."
#             python main.py \
#                 --dataname "$data" \
#                 --model_type $model_type \
#                 --exp_name "$exp_name" \
#                 --hidden_dim "$hidden_dim" \
#                 --learning_rate "$learning_rate" \
#                 --mask_ratio "$mask_ratio" \
#                 --num_eval_repeat "$num_eval_repeat"
#         done
#     done
# done