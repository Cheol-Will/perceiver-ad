#!/usr/bin/env bash

model_type="LATTEMultitask"
learning_rate=0.001
num_heads=4
hidden_dim=64
num_latents=64
num_memories=512
temperature=0.1
sim_type='cos'
sche_gamma=0.98

exp_name="${model_type}_ws1_lr${learning_rate}_h${num_heads}_d${hidden_dim}_lat${num_latents}_mem${num_memories}_temp${temperature}_${sim_type}"

echo "Running experiment: ${exp_name}"
python main_multitask.py \
    --model_type $model_type \
    --exp_name $exp_name \
    --learning_rate $learning_rate \
    --sche_gamma $sche_gamma \
    --num_heads $num_heads \
    --hidden_dim $hidden_dim \
    --num_latents $num_latents \
    --num_memories $num_memories \
    --temperature $temperature \
    --sim_type $sim_type \
    --is_weight_sharing