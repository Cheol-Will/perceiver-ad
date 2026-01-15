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

model_type="TCL"
temperature=0.1
mixup_alpha=1.0

exp_name="$model_type-temp$temperature-mixup_alpha$mixup_alpha"
for data in "${data_list[@]}"; do
    echo "Running $exp_name on $data."
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --temperature "$temperature" \
        --mixup_alpha "$mixup_alpha" \
        --plot_histogram \
        --plot_latent
done