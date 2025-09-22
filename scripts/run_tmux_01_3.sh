#!/bin/bash

# Dataset from MCM and Disent-AD
# model_list=(IForest LOF OCSVM ECOD KNN PCA AutoEncoder DeepSVDD GOAD NeuTraL ICL MCM DRL Disent) 
# data_list=(arrhythmia optdigits breastw cardio campaign cardiotocography census fraud glass ionosphere mammography nslkdd  pendigits pima satellite "satimage-2" shuttle thyroid wbc wine) 
# data_list=(census fraud "satimage-2" shuttle) 

# data_list=(pendigits "satimage-2" breastw pima glass ionosphere wbc wine thyroid arrhythmia cardio cardiotocography mammography) 
data_list=(mammography) 
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
hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemPAE"
latent_ratio=4.0

model_type="MemPAE"
train_ratio_list=(0.2 0.4 0.6 0.8)
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