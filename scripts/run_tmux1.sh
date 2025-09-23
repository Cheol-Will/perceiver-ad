#!/usr/bin/env bash


set -euo pipefail

data_list=(census) # from MCM
#  nslkdd
# fraud 
hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemPAE"
train_ratio_list=(1.0)
depth=6
for data in "${data_list[@]}"; do
    for train_ratio in "${train_ratio_list[@]}"; do
        exp_name="$model_type-ws-pos_query+token-L$depth-d$hidden_dim-lr$learning_rate-t$temperature"
        echo "Running $exp_name on $data."
        python main.py \
            --dataname "$data" \
            --model_type $model_type \
            --is_weight_sharing \
            --use_pos_enc_as_query \
            --use_mask_token \
            --depth "$depth"\
            --hidden_dim "$hidden_dim" \
            --learning_rate "$learning_rate" \
            --temperature "$temperature" \
            --exp_name "$exp_name" \
            --train_ratio "$train_ratio"
    done
done



#!/usr/bin/env bash
set -euo pipefail

data_list=(
    local_anomalies_18_Ionosphere_42
    local_anomalies_45_wine_42
    local_anomalies_14_glass_42
    local_anomalies_42_WBC_42
    local_anomalies_18_Ionosphere_42
    local_anomalies_4_breastw_42
    local_anomalies_29_Pima_42
    local_anomalies_6_cardio_42
    local_anomalies_7_Cardiotocography_42
    local_anomalies_38_thyroid_42
    local_anomalies_26_optdigits_42
    local_anomalies_31_satimage-2_42
    local_anomalies_30_satellite_42
    local_anomalies_23_mammography_42
    local_anomalies_5_campaign_42
    local_anomalies_32_shuttle_42

    cluster_anomalies_18_Ionosphere_42
    cluster_anomalies_45_wine_42
    cluster_anomalies_14_glass_42
    cluster_anomalies_42_WBC_42
    cluster_anomalies_18_Ionosphere_42
    cluster_anomalies_4_breastw_42
    cluster_anomalies_29_Pima_42
    cluster_anomalies_6_cardio_42
    cluster_anomalies_7_Cardiotocography_42
    cluster_anomalies_38_thyroid_42
    cluster_anomalies_26_optdigits_42
    cluster_anomalies_31_satimage-2_42
    cluster_anomalies_30_satellite_42
    cluster_anomalies_23_mammography_42
    cluster_anomalies_5_campaign_42
    cluster_anomalies_32_shuttle_42
)      
   
# data_list=(arrhythmia  cardio cardiotocography pima  optdigits pendigits satellite) # from MCM
# data_list=(breastw glass ionosphere  wbc wine thyroid ) 

hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemPAE"
train_ratio_list=(1.0)
top_k=5
depth=5
for data in "${data_list[@]}"; do
    for train_ratio in "${train_ratio_list[@]}"; do
        exp_name="$model_type-ws-pos_query+token-np-top$top_k-L$depth-d$hidden_dim-lr$learning_rate-t$temperature"
        echo "Running $exp_name on $data."
        python main.py \
            --dataname "$data" \
            --model_type $model_type \
            --is_weight_sharing \
            --use_pos_enc_as_query \
            --use_mask_token \
            --not_use_power_of_two\
            --top_k $top_k\
            --depth $depth\
            --hidden_dim "$hidden_dim" \
            --learning_rate "$learning_rate" \
            --temperature "$temperature" \
            --exp_name "$exp_name" \
            --train_ratio "$train_ratio"
    done
done





# set -euo pipefail
# model_list=(KNN MCM DRL Disent) 
# data_list=(arrhythmia breastw glass cardio cardiotocography pima wbc wine campaign) # from MCM
# depth=4
# hidden_dim=64
# learning_rate=0.001
# temperature=0.1
# contamination_ratio_list=(0.01 0.03 0.05)

# for model_type in "${model_list[@]}"; do
#     for data in "${data_list[@]}"; do
#         for contamination_ratio in "${contamination_ratio_list[@]}"; do
#             echo "$model_type on $data with contamination_ratio=$contamination_ratio"
#             python main.py \
#                 --dataname "$data" \
#                 --model_type $model_type \
#                 --contamination_ratio "$contamination_ratio"
#         done
#     done
# done
