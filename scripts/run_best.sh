
#!/bin/bash

data_list=(census fraud nslkdd shuttle "satimage-2") 
model_type='PAE'
hidden_dim=64
learning_rate=0.001
exp_name="$model_type-ws-d$hidden_dim-lr$learning_rate"


model_type='MemPAE'
hidden_dim=32
exp_name="$model_type-ws-d$hidden_dim-lr$learning_rate"
# MemPAE-ws-d32-lr0.001
for data in "${data_list[@]}"; do
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name"\
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate"\
        --is_weight_sharing
done


model_type='MemPAE'
hidden_dim=16
exp_name="$model_type-ws-d$hidden_dim-lr$learning_rate"
# MemPAE-ws-d32-lr0.001
for data in "${data_list[@]}"; do
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name"\
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate"\
        --is_weight_sharing
done



# PAE-ws-d32-lr0.001
hidden_dim=32
exp_name="$model_type-ws-d$hidden_dim-lr$learning_rate"
for data in "${data_list[@]}"; do
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name"\
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate"\
        --is_weight_sharing
done

# PAE-ws-d16-lr0.001
hidden_dim=16
exp_name="$model_type-ws-d$hidden_dim-lr$learning_rate"
for data in "${data_list[@]}"; do
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name"\
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate"\
        --is_weight_sharing
done

# PAE-ws-d64-lr0.001
for data in "${data_list[@]}"; do
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name"\
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate"\
        --is_weight_sharing
done

model_type='MemPAE'
exp_name="$model_type-ws-d$hidden_dim-lr$learning_rate"
# MemPAE-ws-d64-lr0.001
for data in "${data_list[@]}"; do
    python main.py \
        --dataname "$data" \
        --model_type $model_type \
        --exp_name "$exp_name"\
        --hidden_dim "$hidden_dim" \
        --learning_rate "$learning_rate"\
        --is_weight_sharing
done

