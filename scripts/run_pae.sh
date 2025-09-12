#!/usr/bin/env bash
set -euo pipefail

data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign mammography nslkdd fraud "satimage-2" shuttle census) # from MCM
# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere hepatitis pima wbc wine thyroid optdigits pendigits satellite campaign mammography) # from MCM
# data_list=(dependency_anomalies_32_shuttle_42 global_anomalies_32_shuttle_42 cluster_anomalies_32_shuttle_42 local_anomalies_32_shuttle_42)

train_ratio_list=(1.0)

# perceiver -------------------------
hidden_dim_list=(64)
learning_rate_list=(0.001)
model_type="PAE"
depth=6
for data in "${data_list[@]}"; do
    for hidden_dim in "${hidden_dim_list[@]}"; do
        for learning_rate in "${learning_rate_list[@]}"; do
            echo "Running $model_type data=$data dim=$hidden_dim learning_rate=$learning_rate weight sharing"
            exp_name="$model_type-ws-L$depth-d$hidden_dim-lr$learning_rate"
            python main.py \
                --dataname "$data" \
                --model_type $model_type \
                --exp_name "$exp_name"\
                --hidden_dim "$hidden_dim" \
                --learning_rate "$learning_rate"\
                --depth "$depth"\
                --is_weight_sharing
        done
    done
done

# perceiver -------------------------
hidden_dim_list=(64)
learning_rate_list=(0.001)
model_type="PAE"
depth=2
for data in "${data_list[@]}"; do
    for hidden_dim in "${hidden_dim_list[@]}"; do
        for learning_rate in "${learning_rate_list[@]}"; do
            echo "Running $model_type data=$data dim=$hidden_dim learning_rate=$learning_rate weight sharing"
            exp_name="$model_type-ws-L$depth-d$hidden_dim-lr$learning_rate"
            python main.py \
                --dataname "$data" \
                --model_type $model_type \
                --exp_name "$exp_name"\
                --hidden_dim "$hidden_dim" \
                --learning_rate "$learning_rate"\
                --depth "$depth"\
                --is_weight_sharing
        done
    done
done