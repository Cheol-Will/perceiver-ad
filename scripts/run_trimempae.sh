#!/usr/bin/env bash
set -euo pipefail

# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere hepatitis pima wbc wine thyroid optdigits pendigits satellite campaign mammography nslkdd fraud "satimage-2" shuttle census) # from MCM
# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere hepatitis pima wbc wine thyroid optdigits pendigits satellite campaign mammography "satimage-2") # from MCM
data_list=(nslkdd fraud shuttle census)
# data_list=(dependency_anomalies_32_shuttle_42 global_anomalies_32_shuttle_42 cluster_anomalies_32_shuttle_42 local_anomalies_32_shuttle_42)

train_ratio_list=(1.0)
hidden_dim_list=(64 32)
learning_rate_list=(0.001 0.005)
# model_type="TripletMemPAE"
model_type="PairMemPAE"

for hidden_dim in "${hidden_dim_list[@]}"; do
    for learning_rate in "${learning_rate_list[@]}"; do
        for data in "${data_list[@]}"; do
            echo "Running $model_type data=$data dim=$hidden_dim learning_rate=$learning_rate weight sharing"
            exp_name="$model_type-ws-d$hidden_dim-lr$learning_rate"
            python main.py \
                --dataname "$data" \
                --model_type $model_type \
                --exp_name "$exp_name"\
                --hidden_dim "$hidden_dim" \
                --learning_rate "$learning_rate"\
                --is_weight_sharing
        done
    done
done