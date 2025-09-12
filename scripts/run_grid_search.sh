#!/usr/bin/env bash
set -euo pipefail

# data_list=(arrhythmia breastw cardio census campaign cardiotocography glass ionosphere mammography nslkdd hepatitis optdigits pendigits pima satellite shuttle thyroid wbc wine) # from MCM
# data_list=(arrhythmia breastw cardio campaign cardiotocography glass ionosphere mammography nslkdd hepatitis optdigits pendigits pima satellite satimage-2 shuttle thyroid wbc wine) # from MCM
data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere hepatitis pima wbc wine thyroid optdigits pendigits satellite campaign mammography) # from MCM
train_ratio_list=(1.0)

# mlp -------------------------------
hidden_dim_list=(256 128 64)
# depth=(1 2 3)
# num_repeat_list=(1 3 5)
learning_rate_list=(0.001 0.005 0.01 0.05)

# perceiver -------------------------
# hidden_dim_list=(8 16 32)
# learning_rate_list=(0.001 0.005 0.01 0.05)

# drop_col_prob_list=(0.1 0.3 0.5)

# transformer -----------------------
# hidden_dim_list=(8 16 32)
# learning_rate_list=(0.001 0.005 0.01 0.05)
# num_repeat_list=(1 3 5)

model_type="MemAE"
# model_type="RINMLP"
# model_type="PAE"
temperature=0.1

for data in "${data_list[@]}"; do
    for hidden_dim in "${hidden_dim_list[@]}"; do
        for learning_rate in "${learning_rate_list[@]}"; do
            # for drop_prob in "${drop_col_prob_list[@]}"; do
            # for num_repeat in "${num_repeat_list[@]}"; do
            # echo "Running $model_type $data ratio=$ratio dim=$hidden_dim drop_prob=$drop_prob"
            echo "Running $model_type data=$data dim=$hidden_dim learning_rate=$learning_rate"
            # exp_name="$model_type-d$hidden_dim-lr$learning_rate"
            # exp_name="$model_type-ws-d$hidden_dim-lr$learning_rate"
            exp_name="$model_type-l2-d$hidden_dim-lr$learning_rate-t$temperature"
            python main.py \
                --dataname "$data" \
                --model_type $model_type \
                --exp_name "$exp_name"\
                --hidden_dim "$hidden_dim" \
                --learning_rate "$learning_rate"\
                --temperature "$temperature"\
                --sim_type "l2"
                # --is_weight_sharing
                # --num_repeat "$num_repeat"
                # --drop_col_prob "$drop_prob"
            # done
        done
    done
done