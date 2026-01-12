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

    cardio cardiotocography thyroid optdigits 
    satellite "satimage-2" pendigits

    mammography 
    campaign 
    shuttle 
    fraud 
    nslkdd 
    census
) 

model_type="MOCO"
hidden_dim_list=(32)
hidden_dim_list=(64)
hidden_dim_list=(128)
contrastive_loss_weight_list=(1.0 0.1)
momentum=0.999
mixup_alpha_list=(0.3)
temperature_list=(1.0 0.1) 
learning_rate=0.01

for data in "${data_list[@]}"; do
    for hidden_dim in "${hidden_dim_list[@]}"; do
        for mixup_alpha in "${mixup_alpha_list[@]}"; do
            for contrastive_loss_weight in "${contrastive_loss_weight_list[@]}"; do            
                for temperature in "${temperature_list[@]}"; do
                    # exp_name="$model_type-d$hidden_dim-qs$queue_size-mo$momentum-top_k$top_k-temp$temperature-lr$learning_rate"
                    exp_name="$model_type-d$hidden_dim-mo$momentum-mixup_alpha$mixup_alpha-cont$contrastive_loss_weight-temp$temperature"
                    echo "Running $exp_name on $data."
                    python main.py \
                        --dataname "$data" \
                        --model_type $model_type \
                        --exp_name "$exp_name" \
                        --hidden_dim "$hidden_dim" \
                        --momentum "$momentum" \
                        --mixup_alpha "$mixup_alpha"\
                        --contrastive_loss_weight "$contrastive_loss_weight"\
                        --temperature "$temperature"
                done
            done
        done
    done
done