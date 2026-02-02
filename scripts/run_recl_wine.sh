#!/usr/bin/env bash
set -euo pipefail

data_list=(
  breastw
#   pima
#   wine
  # shuttle
)

model_type="RECLv2"
dacl_alpha=0.95
dacl_beta=0.8
contra_loss_weight=0.05
contra_loss_weight=0.1
bs=64

patience_list=(50)
min_delta_list=(0)

for data in "${data_list[@]}"; do
    for patience in "${patience_list[@]}"; do
        for min_delta in "${min_delta_list[@]}"; do

        exp_name="${model_type}-260202-cw${contra_loss_weight}-ap${dacl_alpha}-bt${dacl_beta}-bs${bs}-pa${patience}-de${min_delta}"

        echo "Running ${exp_name} on ${data}."
        python main.py \
            --dataname "${data}" \
            --model_type "${model_type}" \
            --exp_name "${exp_name}" \
            --contra_loss_weight "${contra_loss_weight}" \
            --dacl_alpha "${dacl_alpha}" \
            --dacl_beta "${dacl_beta}" \
            --batch_size "${bs}" \
            --patience "${patience}" \
            --min_delta "${min_delta}" \
            --runs 5
        done
    done
done