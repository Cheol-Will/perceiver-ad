#!/bin/bash

model_type=PAE
learning_rate=0.001
num_latents=2
hidden_dim=16
exp_name="$model_type-small-d$hidden_dim-lr$learning_rate"

python main.py \
    --dataname "optdigits"\
    --model_type "$model_type"\
    --exp_name "$exp_name"\
    --hidden_dim "$hidden_dim"\
    --learning_rate "$learning_rate"\
    --num_latents "$num_latents"\
    --is_weight_sharing