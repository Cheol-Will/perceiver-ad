#!/usr/bin/env bash
set -euo pipefail

data_list=(
#   wine
#   glass
#   wbc
#   ionosphere
#   arrhythmia
#   breastw
#   pima
#   optdigits
#   cardio
#   cardiotocography
#   thyroid
#   satellite
#   "satimage-2"
#   pendigits
  mammography
  campaign
  shuttle
  # fraud
  # nslkdd
  # census
)

model_type="TAEDACLv6"
contra_loss_weight="0.05"
cycle_loss_weight="0.01"
temperature="0.2"

dacl_alpha_list=(0.95 0.85 0.75)
dacl_beta_list=(0.9 0.8 0.7)

for data in "${data_list[@]}"; do
  for dacl_alpha in "${dacl_alpha_list[@]}"; do
    for dacl_beta in "${dacl_beta_list[@]}"; do

      # Skip (alpha=0.95, beta=0.8)
      if [[ "$dacl_alpha" == "0.95" && "$dacl_beta" == "0.8" ]]; then
        echo "[SKIP] $data alpha=$dacl_alpha beta=$dacl_beta"
        continue
      fi

      exp_name="${model_type}-260201-ap${dacl_alpha}-be${dacl_beta}"
      echo "Running $exp_name on $data (alpha=$dacl_alpha, beta=$dacl_beta)."

      python main.py \
        --dataname "$data" \
        --model_type "$model_type" \
        --exp_name "$exp_name" \
        --contra_loss_weight "$contra_loss_weight" \
        --dacl_alpha "$dacl_alpha" \
        --dacl_beta "$dacl_beta" \
        --cycle_loss_weight "$cycle_loss_weight" \
        --temperature "$temperature" \
        --runs 5
    done
  done
done
