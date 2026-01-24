#!/usr/bin/env bash
set -euo pipefail

data_list=(    
    # wine
    # optdigits 
    # wine 
    # glass 
    # wbc 
    # ionosphere 
    # arrhythmia 
    # breastw 
    # pima  
    # cardio cardiotocography thyroid 
    
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

model_type="TAEDACL"
# byol_loss_weight=0.01
byol_loss_weight=0.1

for data in "${data_list[@]}"; do
    exp_name="$model_type-bw$byol_loss_weight"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --byol_loss_weight "$byol_loss_weight"
done

# for data in "${data_list[@]}"; do
#     for hidden_dim in "${hidden_dim_list[@]}"; do
#         for learning_rate in "${learning_rate_list[@]}"; do
#             exp_name="TAE-tuned"
#             # exp_name="$model_type-d$hidden_dim-lr$learning_rate"
#             echo "Running $exp_name on $data."
#             python main.py \
#                 --dataname "$data" \
#                 --model_type $model_type \
#                 --exp_name "$exp_name"
#                 # --hidden_dim "$hidden_dim" \
#                 # --learning_rate "$learning_rate"
#         done
#     done
# done