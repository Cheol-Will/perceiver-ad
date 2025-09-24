

#!/usr/bin/env bash
set -euo pipefail

# data_list=("satimage-2" mammography campaign shuttle nslkdd fraud census)   
data_list=( fraud nslkdd ) # from MCM

hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemPAE"

for data in "${data_list[@]}"; do
    exp_name="$model_type-ws-pos_query+token-mlp_enc_mixer-d$hidden_dim-lr$learning_rate-t$temperature"
    echo "Running $exp_name on $data."
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --is_weight_sharing \
        --use_pos_enc_as_query \
        --use_mask_token \
        --mlp_mixer_encoder\
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --temperature "$temperature" \
        --exp_name "$exp_name"
done



# data_list=(
#     # thyroid
#     # optdigits
#     # "satimage-2"
#     # satellite
#     # campaign
#     # nslkdd
#     # fraud
#     # cardio
#     # cardiotocography
#     # mammography
#     census
# ) 
# model_list=(DRL MCM Disent) 

# train_ratio_list=(0.6 0.8)
# for train_ratio in "${train_ratio_list[@]}"; do
#     for data in "${data_list[@]}"; do
#         for model_type in "${model_list[@]}"; do
#             python main.py \
#                 --dataname "$data" \
#                 --model_type $model_type \
#                 --train_ratio "$train_ratio"
#         done
#     done
# done




# #!/usr/bin/env bash
# set -euo pipefail

# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite mammography "satimage-2" campaign census fraud nslkdd shuttle census fraud nslkdd shuttle) # from MCM
# # data_list=(census fraud )
# # data_list=(nslkdd shuttle)
# hidden_dim=64
# learning_rate=0.001
# temperature=0.1
# model_type="MemPAE"
# train_ratio_list=(1.0)
# for data in "${data_list[@]}"; do
#     for train_ratio in "${train_ratio_list[@]}"; do
#         exp_name="$model_type-ws-pos_query+token-mlp_dec-d$hidden_dim-lr$learning_rate-t$temperature"
#         echo "Running $exp_name on $data."
#         python main.py \
#             --dataname "$data" \
#             --model_type $model_type \
#             --is_weight_sharing \
#             --use_pos_enc_as_query \
#             --use_mask_token \
#             --mlp_decoder \
#             --hidden_dim "$hidden_dim" \
#             --learning_rate "$learning_rate" \
#             --exp_name "$exp_name" \
#             --train_ratio "$train_ratio"
#     done
# done
