#!/usr/bin/env bash
set -euo pipefail

data_list=(
    # arrhythmia breastw cardio cardiotocography glass ionosphere # done
    # pima wbc wine thyroid # done
    # optdigits pendigits satellite # done
    # "satimage-2" campaign # ing
    # mammography shuttle nslkdd fraud census # ing
) # from MCM
learning_rate=0.001
train_ratio=(1.0)
hidden_dim_list=(32 64)
k_list=(1 3 5 10)
temperature_list=(0.1 1.0)
model_type="MemPAE"

for data in "${data_list[@]}"; do
    for hidden_dim in "${hidden_dim_list[@]}"; do
        for k in "${k_list[@]}"; do
            for temperature in "${temperature_list[@]}"; do
                exp_name="$model_type-ws-l2-local+global-sqrt_F-sqrt_N-top$k-d$hidden_dim-lr$learning_rate-t$temperature"
                echo "Running $exp_name on $data."
                python main.py \
                    --dataname "$data" \
                    --model_type $model_type \
                    --is_weight_sharing \
                    --sim_type 'l2' \
                    --use_mask_token \
                    --use_pos_enc_as_query \
                    --top_k "$k"\
                    --hidden_dim "$hidden_dim" \
                    --learning_rate "$learning_rate" \
                    --temperature "$temperature" \
                    --exp_name "$exp_name"
            done
        done
    done
done