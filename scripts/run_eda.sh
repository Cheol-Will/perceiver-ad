#!/usr/bin/env bash
set -euo pipefail

data_list=(   
    campaign
    shuttle 
    thyroid 
    optdigits 
    wbc
    glass
    pima
    breastw    
    wine
    cardiotocography
    cardio 
    satellite 
    pendigits
    arrhythmia 
    thyroid 
    ionosphere 
    mammography 
    census
    fraud
    shuttle
) 

for data in "${data_list[@]}"; do
    python eda.py \
        --dataname "$data" \
        --dist \
        --box \
        --tsne \
        --umap
done