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
dacl_alpha=0.9
byol_loss_weight_list=(0.1)
# byol_loss_weight_list=(0.01)
epochs=200
for data in "${data_list[@]}"; do
    for byol_loss_weight in "${byol_loss_weight_list[@]}"; do
        # exp_name="$model_type-260125-bw$byol_loss_weight-ap$dacl_alpha-ep$epochs"
        exp_name="$model_type-260125-bw$byol_loss_weight-ap$dacl_alpha"
        echo "Running $exp_name on $data."
        python main.py \
            --dataname "$data" \
            --model_type $model_type \
            --exp_name "$exp_name" \
            --byol_loss_weight "$byol_loss_weight" \
            --dacl_alpha $dacl_alpha
            # --epochs $epochs
    done
done