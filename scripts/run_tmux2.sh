#!/usr/bin/env bash
set -euo pipefail

data_list=(
    contamination_0.01_18_Ionosphere_42
    contamination_0.03_18_Ionosphere_42
    contamination_0.05_18_Ionosphere_42
    contamination_0.01_32_shuttle_42
    contamination_0.03_32_shuttle_42
    contamination_0.05_32_shuttle_42
    contamination_0.01_38_thyroid_42
    contamination_0.03_38_thyroid_42
    contamination_0.05_38_thyroid_42
    contamination_0.01_26_optdigits_42
    contamination_0.03_26_optdigits_42
    contamination_0.05_26_optdigits_42
    contamination_0.01_23_mammography_42
    contamination_0.03_23_mammography_42
    contamination_0.05_23_mammography_42
)
data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign mammography "satimage-2" nslkdd fraud  shuttle census) # from MCM
sim_type='cross_attn'
entropy_loss_weight=0.01
depth=4
hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemPAE"

for data in "${data_list[@]}"; do
    exp_name="$model_type-ws-$sim_type-pos_query+token-use_ent_score-ent$entropy_loss_weight-d$hidden_dim-lr$learning_rate"
    echo "$exp_name on $data"
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --is_weight_sharing \
        --sim_type "$sim_type" \
        --use_pos_enc_as_query \
        --use_mask_token \
        --use_entropy_loss_as_score \
        --entropy_loss_weight "$entropy_loss_weight" \
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate" \
        --exp_name "$exp_name"
done