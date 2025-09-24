#!/bin/bash
data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign mammography) # from MCM

data_list=(
    # breastw # best SHAP vs Attn
    # glass
    # wine
    # wbc
    # "satimage-2"
    thyroid 
    ionosphere
    pendigits
    cardio
    mammography
    pima
    cardiotocography


    # fraud
    # mammography
    # pendigits
    # census
    # nslkdd

    # fraud
    # nslkdd
    # census
    ###
    # arrhythmia
    # cardiotocography
    # optdigits 
    # pendigits 
    # satellite 
    # campaign 
    # wine
    # pima
    # cardio 
    # glass 
    # ionosphere
    # "satimage-2"
    # satellite
    # shuttle
    # cardio
    # wbc
    # thyroid
    ###
)
model_type='MemPAE'
hidden_dim=64
learning_rate=0.001
temperature=0.1
for data in "${data_list[@]}"; do
    exp_name="$model_type-ws-pos_query+token-d$hidden_dim-lr$learning_rate-t$temperature"
    echo "$exp_name on $data"
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --is_weight_sharing \
        --use_pos_enc_as_query \
        --use_mask_token \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --temperature "$temperature" \
        --exp_name "$exp_name" \
        --compare_shap_vs_encoder_attention_per_sample
        # --visualize_attention_vs_shap
done