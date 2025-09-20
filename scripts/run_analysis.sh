#!/bin/bash

# data_list=(satimage-2) # from MCM
data_list=(pendigits optdigits arrhythmia breastw glass ionosphere pima wbc wine cardio cardiotocography thyroid) # from MCM

data_list=(arrhythmia breastw cardio cardiotocography glass ionosphere pima wbc wine thyroid optdigits pendigits satellite campaign mammography nslkdd shuttle fraud census) # from MCM
# data_list=(pendigits) # from MCM
# data_list=(cardiotocography) # from MCM

# data_list=(pima) # from MCM
# data_list=(wine wbc) # from MCM
# data_list=(cardiotocography) # from MCM
# data_list=(
#     # dependency_anomalies_45_wine_42
#     # dependency_anomalies_14_glass_42
#     # dependency_anomalies_42_WBC_42
#     # dependency_anomalies_18_Ionosphere_42
#     # dependency_anomalies_4_breastw_42
#     # dependency_anomalies_29_Pima_42
#     # dependency_anomalies_6_cardio_42
#     # dependency_anomalies_7_Cardiotocography_42
#     # dependency_anomalies_38_thyroid_42
#     # dependency_anomalies_26_optdigits_42
#     # dependency_anomalies_31_satimage-2_42
#     # dependency_anomalies_30_satellite_42
#     # dependency_anomalies_23_mammography_42
#     # dependency_anomalies_5_campaign_42
#     # dependency_anomalies_32_shuttle_42
#     mnist
# )
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
        --exp_name "$exp_name"\
        --plot_pos_encoding
        # --plot_histogram
        # --plot_recon
        # --plot_tsne_recon
done