#!/usr/bin/env bash
set -euo pipefail

data_list=(    
    # wine
    # glass 
    # wbc 
    # ionosphere 
    # arrhythmia 
    # breastw 
    pima  
    # optdigits 
    
    # cardio 
    # cardiotocography 
    # thyroid 
    # satellite 
    # "satimage-2" 
    # pendigits
    # mammography 
    # campaign 
    # shuttle 
    
    # fraud 
    # nslkdd 
    # census
) 

model_type="TAEDACLv6"
dacl_alpha=0.95
dacl_beta=0.8

contra_loss_weight=0.05

cycle_loss_weight=0.01
# cycle_loss_weight=0.02
# cycle_loss_weight=0.005

bs=128
bs=64
# bs=256
# bs=512

temperature=0.1
temperature=0.2
 
epochs=50
epochs_list=(50 100)
epochs_list=(20)
for data in "${data_list[@]}"; do
        for epochs in "${epochs_list[@]}"; do

        exp_name="$model_type-260131-cw$contra_loss_weight-ap$dacl_alpha-bt$dacl_beta-cycle$cycle_loss_weight-bs$bs-temp$temperature-ep$epochs"
        # exp_name="$model_type-260131-cw$contra_loss_weight-ap$dacl_alpha-bt$dacl_beta-cycle$cycle_loss_weight-bs$bs-temp$temperature"
        echo "Running $exp_name on $data."
        python main.py \
            --dataname "$data" \
            --model_type $model_type \
            --exp_name "$exp_name" \
            --contra_loss_weight "$contra_loss_weight" \
            --dacl_alpha $dacl_alpha \
            --dacl_beta $dacl_beta \
            --cycle_loss_weight $cycle_loss_weight \
            --batch_size $bs \
            --temperature $temperature \
            --epochs $epochs \
            --runs 5
    done
done