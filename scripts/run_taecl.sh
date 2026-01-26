#!/usr/bin/env bash
set -euo pipefail

data_list=(    
    wine 
    # glass 
    # wbc 
    # ionosphere 
    # arrhythmia 
    # breastw 
    # pima  

    # cardio 
    # cardiotocography

    #  thyroid 
    # optdigits 
    # "satimage-2" 
    # satellite 
    
    # pendigits
    # mammography 
    # campaign 

    shuttle 
    fraud 

    # nslkdd 
    # census
) 

model_type="TAECL"
temperature_list=(0.2)
# temperature_list=(0.1)
# temperature_list=(1.0)
contrastive_loss_weight_list=(0.01)
# epoch_list=(10 30 50 100 200)
contrastive_loss_weight=0.01
batch_size_list=(128 256)
for data in "${data_list[@]}"; do
    for temperature in "${temperature_list[@]}"; do
        for batch_size in "${batch_size_list[@]}"; do
            exp_name="TAECL-250124-bs$batch_size"
            # exp_name="$model_type-temp$temperature-contra$contrastive_loss_weight"
            echo "Running $exp_name on $data."
            python main.py \
                --dataname "$data" \
                --model_type $model_type \
                --exp_name "$exp_name" \
                --temperature "$temperature" \
                --contrastive_loss_weight "$contrastive_loss_weight" \
                --batch_size $batch_size
        done
    done
done