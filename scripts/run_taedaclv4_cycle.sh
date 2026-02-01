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
    fraud #########
    nslkdd #########
    census ##########
) 

model_type="TAEDACLv4"
dacl_alpha=0.95
dacl_beta=0.8
# cycle_loss_weight_list=(0.01 0.02 0.05 0.001 0.002 0.005)
# cycle_loss_weight_list=(0.007 0.005 0.003 0.002 0.001)

##############################################
##############################################
# contra_loss_weight_list=(0.01)
contra_loss_weight_list=(0.1 0.05 0.01)
contra_loss_weight_list=(0.1)
contra_loss_weight_list=(0.5 0.1)
# contra_loss_weight_list=(0.01)
##############################################

contra_loss_weight=0.05
cycle_loss_weight=0.001

contra_loss_weight=0.05
# cycle_loss_weight=0.002
cycle_loss_weight=0.005
cycle_loss_weight=0.0007

for data in "${data_list[@]}"; do
    exp_name="$model_type-260130-cw$contra_loss_weight-ap$dacl_alpha-bt$dacl_beta-cycle$cycle_loss_weight-bs-tuned"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --contra_loss_weight "$contra_loss_weight" \
        --dacl_alpha $dacl_alpha \
        --dacl_beta $dacl_beta \
        --cycle_loss_weight $cycle_loss_weight \
        --config_file_name "TAEDACLv4_Cycle" \
        --runs 5
done


# for data in "${data_list[@]}"; do
#     for contra_loss_weight in "${contra_loss_weight_list[@]}"; do
#         for cycle_loss_weight in "${cycle_loss_weight_list[@]}"; do
#             exp_name="$model_type-260130-cw$contra_loss_weight-ap$dacl_alpha-bt$dacl_beta-cycle$cycle_loss_weight"
#             echo "Running $exp_name on $data."
#             python main.py \
#                 --dataname "$data" \
#                 --model_type $model_type \
#                 --exp_name "$exp_name" \
#                 --contra_loss_weight "$contra_loss_weight" \
#                 --dacl_alpha $dacl_alpha \
#                 --dacl_beta $dacl_beta \
#                 --cycle_loss_weight $cycle_loss_weight \
#                 --runs 5
#         done
#     done
# done