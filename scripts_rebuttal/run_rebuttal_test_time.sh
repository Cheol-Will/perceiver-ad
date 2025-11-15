#!/usr/bin/env bash
set -euo pipefail

data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite "satimage-2" campaign mammography shuttle nslkdd fraud census) # from MCM
hidden_dim=64
learning_rate=0.001
temperature=0.1
memory_ratio=1.0
latent_ratio=1.0
model_type="MemPAE"

# for data in "${data_list[@]}"; do
#     exp_name="$model_type-test_time-ws-local+global-sqrt_F$latent_ratio-sqrt_N$memory_ratio-d$hidden_dim-lr$learning_rate-t$temperature"
#     python main.py \
#         --runs 1 \
#         --dataname "$data" \
#         --model_type $model_type \
#         --is_weight_sharing \
#         --use_mask_token \
#         --use_pos_enc_as_query \
#         --latent_ratio $latent_ratio \
#         --memory_ratio $memory_ratio \
#         --hidden_dim "$hidden_dim" \
#         --learning_rate "$learning_rate" \
#         --temperature "$temperature" \
#         --exp_name "$exp_name"
# done


model_type="KNN"
for data in "${data_list[@]}"; do
    exp_name="$model_type-test_time"
    python main.py \
        --runs 1 \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name"
done

# data_list=(
#     shuttle
#     "satimage-2"
#     satellite
#     ionosphere
#     wbc
#     cardiotocography
#     pendigits 
#     cardio
#     mammography 
#     wine 
#     glass 
#     breastw 
#     pima 
#     thyroid 
# )

# model_type="NPTAD"
# for data in "${data_list[@]}"; do
#     exp_name="$model_type-test_time"
#     python main.py \
#         --runs 1 \
#         --dataname "$data" \
#         --model_type $model_type \
#         --exp_name "$exp_name"
# done
