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
    # cardio cardiotocography thyroid 
    
    optdigits 
    satellite "satimage-2" pendigits

    mammography 
    campaign 
    shuttle 
    fraud 
    nslkdd 
    census
) 

model_type="ProtoAD"
hidden_dim_list=(32)
num_prototypes_list=(32 64 128)
sinkhorn_eps_list=(0.05)
contrastive_loss_weight_list=(0.1)
temperature_list=(0.1) 
learning_rate=0.01

for data in "${data_list[@]}"; do
    for hidden_dim in "${hidden_dim_list[@]}"; do
        for num_prototypes in "${num_prototypes_list[@]}"; do
            for sinkhorn_eps in "${sinkhorn_eps_list[@]}"; do            
                for contrastive_loss_weight in "${contrastive_loss_weight_list[@]}"; do
                    for temperature in "${temperature_list[@]}"; do
                        exp_name="$model_type-d$hidden_dim-proto$num_prototypes-eps$sinkhorn_eps-contra$contrastive_loss_weight-temp$temperature"
                        echo "Running $exp_name on $data."
                        python main.py \
                            --dataname "$data" \
                            --model_type $model_type \
                            --exp_name "$exp_name" \
                            --hidden_dim "$hidden_dim" \
                            --num_prototypes "$num_prototypes" \
                            --sinkhorn_eps "$sinkhorn_eps" \
                            --contrastive_loss_weight "$contrastive_loss_weight" \
                            --temperature "$temperature"
                    done
                done
            done
        done
    done
done