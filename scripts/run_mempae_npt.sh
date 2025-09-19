#!/usr/bin/env bash
set -euo pipefail

data_list=(
    # lympho
    # vertebral
    # vowels
    # letter
    # musk
    # speech

    # mnist
    # annthyroid
    forest_cover
    backdoor
    seismic
    # abalone # need to resolve things
    # mulcross # arff
    # ecoli # need to create label manually.
)
  
hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemPAE"
for data in "${data_list[@]}"; do
    exp_name="$model_type-ws-pos_query+token-d$hidden_dim-lr$learning_rate-t$temperature"
    echo "$exp_name on $data"
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --is_weight_sharing \
        --use_pos_enc_as_query \
        --use_mask_token \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --temperature "$temperature" \
        --exp_name "$exp_name"
done