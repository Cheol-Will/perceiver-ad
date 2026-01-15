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

model_type="TAE"
exp_name="TAE-tuned"
for data in "${data_list[@]}"; do
    echo "Running $exp_name on $data."
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name" \
        --plot_histogram
        # --plot_latent
done