#!/bin/bash

# data_list=(satimage-2) # from MCM
# data_list=(arrhythmia breastw glass ionosphere pima wbc wine cardio cardiotocography thyroid pendigits optdigits ) # from MCM
# data_list=(pima) # from MCM
# data_list=(wine wbc) # from MCM
# data_list=(cardiotocography) # from MCM
data_list=(
    dependency_anomalies_45_wine_42
    dependency_anomalies_14_glass_42
    dependency_anomalies_42_WBC_42
    dependency_anomalies_18_Ionosphere_42
    dependency_anomalies_4_breastw_42
    dependency_anomalies_29_Pima_42
    dependency_anomalies_6_cardio_42
    dependency_anomalies_7_Cardiotocography_42
    dependency_anomalies_38_thyroid_42
    dependency_anomalies_26_optdigits_42
    dependency_anomalies_31_satimage-2_42
    dependency_anomalies_30_satellite_42
    dependency_anomalies_23_mammography_42
    dependency_anomalies_5_campaign_42
    dependency_anomalies_32_shuttle_42
) 
# model_list=(IForest LOF OCSVM ECOD KNN PCA MCM DRL Disent) 
# model_type='Disent'
model_type='MCM'
# model_type='PAE'
for data in "${data_list[@]}"; do
    echo "$model_type on $data"
    python analysis.py \
        --dataname "$data" \
        --model_type $model_type \
        --plot_histogram
done