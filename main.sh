#!/usr/bin/env bash
set -euo pipefail

# data_list=(arrhythmia breastw Hepatitis optdigits thyroid)
data_list=(annthyroid breastw mammography Pima thyroid)

train_ratio_list=(0.1 0.5 1.0)
hidden_dims=(8 16 32 64)
drop_col_probs=(0.1 0.3 0.5)

model_type=Perceiver
num_heads=4
num_layers=4
mlp_ratio=4.0
dropout_prob=0.0

fmt() { printf "%s" "${1/./_}"; }

for data in "${data_list[@]}"; do
  for ratio in "${train_ratio_list[@]}"; do
    for d in "${hidden_dims[@]}"; do
      for dcol in "${drop_col_probs[@]}"; do
        drp_tag=$(fmt "$dropout_prob")
        dcol_tag=$(fmt "$dcol")
        exp_name="Perceiver-h${num_heads}-d${d}-L${num_layers}-m${mlp_ratio}-dr${drp_tag}-dcol${dcol_tag}"

        python main.py \
          --dataname "$data" \
          --model_type "$model_type" \
          --train_ratio "$ratio" \
          --num_heads "$num_heads" \
          --num_layers "$num_layers" \
          --hidden_dim "$d" \
          --mlp_ratio "$mlp_ratio" \
          --dropout_prob "$dropout_prob" \
          --drop_col_prob "$dcol" \
          --exp_name "$exp_name"
      done
    done
  done
done
