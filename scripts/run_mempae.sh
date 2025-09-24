#!/usr/bin/env bash
set -euo pipefail

# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite) # from MCM
# data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign mammography ) # from MCM
data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign mammography ) # from MCM
data_list=(
    cardio
    cardiotocography
    mammography
    pendigits
    pima
    wbc
    wine
)

hidden_dim=64
learning_rate=0.001
temperature=0.1
model_type="MemPAE"
# latent_ratio=4.0
train_ratio_list=(1.0)
entropy_loss_weight=3

for data in "${data_list[@]}"; do
    for train_ratio in "${train_ratio_list[@]}"; do
        exp_name="$model_type-ws-pos_query+token-use_ent_score-ent$entropy_loss_weight-d$hidden_dim-lr$learning_rate-t$temperature"
        # exp_name="$model_type-ws-pos_query+token-mlp_dec-d$hidden_dim-lr$learning_rate-t$temperature"
        echo "Running $exp_name on $data."
        python main.py \
            --dataname "$data" \
            --model_type $model_type \
            --is_weight_sharing \
            --use_pos_enc_as_query \
            --use_mask_token \
            --use_entropy_loss_as_score \
            --entropy_loss_weight "$entropy_loss_weight"\
            --hidden_dim "$hidden_dim" \
            --learning_rate "$learning_rate" \
            --temperature "$temperature" \
            --exp_name "$exp_name" \
            --train_ratio "$train_ratio"
    done
done
            # --mlp_mixer_decoder \
