#!/usr/bin/env bash
set -euo pipefail

# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite "satimage-2" campaign mammography shuttle nslkdd fraud census) # from MCM
# num_features up to 30 is fine.
# from cardio ours is faster.
data_list=(
    # fraud
    census

    # arrhythmia
    # optdigits
    # pendigits
    # campaign 

    # nslkdd


    # shuttle
    # "satimage-2"
    # satellite
    # ionosphere
    # wbc
    # cardiotocography
    # pendigits 
    # cardio
    # mammography 
    # wine 
    # glass 
    # breastw 
    # pima 
    # thyroid 
) # from MCM

train_ratio=(1.0)
model="NPTAD"
for data in "${data_list[@]}"; do
    torchrun --nproc_per_node=6 main.py \
        --dataname "$data" \
        --model $model 
done