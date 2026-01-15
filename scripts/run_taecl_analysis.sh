#!/usr/bin/env bash
set -euo pipefail

data_list=(    
    optdigits
)
  
data_list=(    
    # good 
    optdigits 
    cardio 
    satellite 
    pendigits
    
    # not good
    wine
    campaign

    arrhythmia 
    thyroid 
    ionosphere 
    mammography 
    # shuttle # 1 minutes
    # census
) 

model_type="TAECL"
exp_name="TAECL-tuned"
temperature=0.1
contrastive_loss_weight=0.1
exp_name="$model_type-temp$temperature-contra$contrastive_loss_weight"

for data in "${data_list[@]}"; do
    echo "Running $exp_name on $data."
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --temperature "$temperature" \
        --contrastive_loss_weight "$contrastive_loss_weight" \
        --plot_histogram
        # --plot_latent
done