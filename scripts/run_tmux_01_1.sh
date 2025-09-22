#!/bin/bash

#!/usr/bin/env bash
set -euo pipefail

# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite) # from MCM
# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign mammography ) # from MCM
# data_list=("satimage-2" nslkdd fraud)
# data_list=(census)
data_list=(mammography campaign fraud nslkdd pendigits sattelite shuttle "satimage-2" )


# data_list=(
#     global_anomalies_45_wine_42
#     global_anomalies_14_glass_42
#     global_anomalies_42_WBC_42
#     global_anomalies_18_Ionosphere_42
#     global_anomalies_4_breastw_42
#     global_anomalies_29_Pima_42
#     global_anomalies_6_cardio_42
#     global_anomalies_7_Cardiotocography_42
#     global_anomalies_38_thyroid_42
#     global_anomalies_26_optdigits_42
#     global_anomalies_31_satimage-2_42
#     global_anomalies_30_satellite_42
#     global_anomalies_23_mammography_42
#     global_anomalies_5_campaign_42
#     global_anomalies_32_shuttle_42
# )      
   
# data_list=(arrhythmia  cardio cardiotocography pima  optdigits pendigits satellite) # from MCM
# data_list=(breastw glass ionosphere  wbc wine thyroid ) 

hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemPAE"
# latent_ratio=4.0
train_ratio_list=(1.0)
top_k=1
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



# Dataset from MCM and Disent-AD
# model_list=(IForest LOF OCSVM ECOD KNN PCA AutoEncoder DeepSVDD GOAD NeuTraL ICL MCM DRL Disent) 
# data_list=(arrhythmia optdigits breastw cardio campaign cardiotocography census fraud glass ionosphere mammography nslkdd  pendigits pima satellite "satimage-2" shuttle thyroid wbc wine) 
# data_list=(census fraud "satimage-2" shuttle) 

# data_list=(census) 
# data_list=(pendigits "satimage-2" breastw pima glass ionosphere wbc wine thyroid arrhythmia cardio cardiotocography mammography) 
# data_list=(
#     arrhythmia
#     pima
#     pendigits
#     ionosphere
#     "satimage-2" 
#     wbc
#     wine
#     thyroid
#     breastw
#     glass
#     shuttle
# )

# model_list=(KNN Disent MCM DRL) 
# # train_ratio_list=(0.2)
# train_ratio_list=(0.4)
# # train_ratio_list=(0.6)

# for data in "${data_list[@]}"; do
#     for model_type in "${model_list[@]}"; do
#         for train_ratio in "${train_ratio_list[@]}"; do
#             python main.py \
#                 --dataname "$data" \
#                 --model_type "$model_type" \
#                 --exp_name "$model_type" \
#                 --train_ratio "$train_ratio"
#         done
#     done
# done