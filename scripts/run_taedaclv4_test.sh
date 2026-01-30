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
    nslkdd 
    census
) 
model_type="TAEDACLv4"
exp_name="TAEDACLv4-260130-2-cw0.1-ap0.95-bt0.8-ret"
pth_dir_name="TAEDACLv4-260126-cw0.1-ap0.95-bt0.8"
for data in "${data_list[@]}"; do
    echo "Running $exp_name on $data"
    python test.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name $exp_name \
        --pth_dir_name $pth_dir_name \
        --runs 5
done

python results_helper/parse.py --target $exp_name