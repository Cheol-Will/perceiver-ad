#!/usr/bin/env bash
set -euo pipefail

data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite "satimage-2" campaign mammography shuttle nslkdd fraud census) # from MCM
train_ratio=(1.0)
hidden_dim=64
learning_rate=0.001
# temperature_list=(0.1 0.5 1.0)
temperature_list=(0.5)
model_type="MemPAE"
# memory_ratio_list=(0.5 1.0 2.0 4.0)
memory_ratio_list=(2.0 4.0)
latent_ratio_list=(0.5 1.0 2.0 4.0)

for data in "${data_list[@]}"; do
    for memory_ratio in "${memory_ratio_list[@]}"; do
        for latent_ratio in "${latent_ratio_list[@]}"; do
            for temperature in "${temperature_list[@]}"; do
                exp_name="$model_type-ws-local+global-sqrt_F$latent_ratio-sqrt_N$memory_ratio-d$hidden_dim-lr$learning_rate-t$temperature"
                if [[ "$memory_ratio" == "1.0" && "$latent_ratio" == "1.0" ]]; then
                    echo "Coyp since memory_ratio=$memory_ratio and latent_ratio=$latent_ratio."
                    
                    mkdir -p "results/$exp_name"
                    rsync -a "results/$model_type-ws-local+global-sqrt_F-sqrt_N-d$hidden_dim-lr$learning_rate-t$temperature/" \
                            "results/$exp_name/"
                    continue
                fi
                echo "Running $exp_name on $data."
                python main.py \
                    --dataname "$data" \
                    --model_type $model_type \
                    --is_weight_sharing \
                    --use_mask_token \
                    --use_pos_enc_as_query \
                    --latent_ratio $latent_ratio \
                    --memory_ratio $memory_ratio \
                    --hidden_dim "$hidden_dim" \
                    --learning_rate "$learning_rate" \
                    --temperature "$temperature" \
                    --exp_name "$exp_name"
            done
        done
    done
done