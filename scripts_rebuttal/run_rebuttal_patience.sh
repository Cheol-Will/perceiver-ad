#!/usr/bin/env bash
set -euo pipefail

data_list=(
    # arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid census
    # optdigits pendigits satellite "satimage-2" campaign fraud
    # mammography 
    # shuttle 
    # nslkdd 
    # 
    # fraud
    # wine glass wbc ionosphere 
    # arrhythmia breastw pima

    # optdigits  nslkdd
    # fraud
    # satellite pendigits mammography campaign 
    # shuttle census 
    # breastw pima cardio cardiotocography thyroid "satimage-2"
    # nslkdd
    # pendigits
    # census
    # mammography
    # campaign
    # shuttle
    # fraud
    # optdigits 
    satellite
    # pendigits  mammography  campaign  shuttle  nslkdd   fraud  census
    # wine glass wbc ionosphere arrhythmia breastw 
    # pima cardio cardiotocography 
    # thyroid "satimage-2"

) # from MCM
model_type="MemPAE"
latent_ratio=1.0
memory_ratio=1.0
hidden_dim=64
learning_rate=0.001
temperature=0.1
patience=20
# delta=0.002
delta_list=(
    0.000001
    # 0.004
    # 0.00001
    # 0.00002
    # 0.0001
    # 0.0002
    # 0.0005
    # 0.001
    # 0.002
    # 0.005
    # 0.01
)
for delta in "${delta_list[@]}"; do
    for data in "${data_list[@]}"; do
        exp_name="LATTE-patience$patience-delta$delta"
        echo "Running $exp_name on $data."
        python main.py \
            --dataname "$data" \
            --model_type $model_type \
            --use_mask_token \
            --use_pos_enc_as_query \
            --is_weight_sharing \
            --latent_ratio $latent_ratio \
            --memory_ratio $memory_ratio \
            --use_num_latents_power_2 \
            --use_num_memories_power_2 \
            --hidden_dim "$hidden_dim" \
            --learning_rate "$learning_rate" \
            --temperature "$temperature" \
            --patience "$patience" \
            --min_delta "$delta" \
            --exp_name "$exp_name" 
    done
done