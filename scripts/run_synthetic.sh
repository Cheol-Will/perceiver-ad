#!/bin/bash
data_list=(
    # cluster_anomalies_45_wine_42
    # cluster_anomalies_14_glass_42
    # cluster_anomalies_42_WBC_42
    # cluster_anomalies_18_Ionosphere_42
    # cluster_anomalies_4_breastw_42
    # cluster_anomalies_29_Pima_42
    # cluster_anomalies_6_cardio_42
    # cluster_anomalies_7_Cardiotocography_42
    # cluster_anomalies_38_thyroid_42
    # cluster_anomalies_26_optdigits_42
    # cluster_anomalies_31_satimage-2_42
    # cluster_anomalies_30_satellite_42
    # cluster_anomalies_23_mammography_42
    # cluster_anomalies_5_campaign_42
    # cluster_anomalies_32_shuttle_42
    global_anomalies_32_shuttle_42
      
   
   
    ##################################
    # local_anomalies_45_wine_42
    # local_anomalies_14_glass_42
    # local_anomalies_42_WBC_42
    # local_anomalies_18_Ionosphere_42
    # local_anomalies_4_breastw_42
    # local_anomalies_29_Pima_42
    # local_anomalies_6_cardio_42
    # local_anomalies_7_Cardiotocography_42
    # local_anomalies_38_thyroid_42
    # local_anomalies_26_optdigits_42
    # local_anomalies_31_satimage-2_42
    # local_anomalies_30_satellite_42
    # local_anomalies_23_mammography_42
    # local_anomalies_5_campaign_42
    # local_anomalies_32_shuttle_42
    ##################################

    ##################################
    # dependency_anomalies_45_wine_42
    # dependency_anomalies_14_glass_42
    # dependency_anomalies_42_WBC_42
    # dependency_anomalies_18_Ionosphere_42
    # dependency_anomalies_4_breastw_42
    # dependency_anomalies_29_Pima_42
    # dependency_anomalies_6_cardio_42
    # dependency_anomalies_7_Cardiotocography_42
    # dependency_anomalies_38_thyroid_42
    # dependency_anomalies_26_optdigits_42
    # dependency_anomalies_31_satimage-2_42
    # dependency_anomalies_30_satellite_42
    # dependency_anomalies_23_mammography_42
    # dependency_anomalies_5_campaign_42
    # dependency_anomalies_32_shuttle_42
    ##################################
)

model_list=(IForest LOF OCSVM ECOD KNN PCA MCM DRL Disent) 
for data in "${data_list[@]}"; do
    for model_type in "${model_list[@]}"; do
        echo "$model_type on $data";
        python main.py \
            --dataname "$data"\
            --model_type "$model_type"\
            --exp_name "$model_type"
    done
done
