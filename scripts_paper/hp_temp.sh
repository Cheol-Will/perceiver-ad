#!/usr/bin/env bash
set -euo pipefail

data_list=(
  wine
  glass
  wbc
  ionosphere
  arrhythmia
  breastw
  pima
  optdigits
  cardio
  cardiotocography
  thyroid
  satellite
  "satimage-2"
  pendigits
  mammography
  campaign
  shuttle
  # fraud
  # nslkdd
  # census
)

model_type="TAEDACLv6"
dacl_alpha="0.95"
dacl_beta="0.8"
contra_loss_weight="0.05"
cycle_loss_weight="0.01"

temperature_list=(0.5 0.2 0.1 0.05)


for data in "${data_list[@]}"; do
      for temperature in "${temperature_list[@]}"; do
        # Skip (temp=0.2, contra=0.05, cycle=0.01)
        if [[ "$temperature" == "0.2" && "$contra_loss_weight" == "0.05" && "$cycle_loss_weight" == "0.01" ]]; then
            echo "[SKIP] $data temp=$temperature contra=$contra_loss_weight cycle=$cycle_loss_weight"
            continue
        fi

        exp_name="${model_type}-260201-t${temperature}-clw${contra_loss_weight}-cyw${cycle_loss_weight}"
        echo "Running $exp_name on $data (temp=$temperature)."

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