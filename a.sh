model_list=(
    'IForest' 
    'LOF' 
    'OCSVM' 
    'ECOD' 
    'KNN' 
    'PCA'
    'AutoEncoder'
    'DeepSVDD'
    'GOAD'
    'NeuTraL'
    'ICL' 
    'MCM'
    'DRL'
    'Disent'
    'PDRL-ws-pos_query+token-d64-lr0.001'
    'PAE-ws-L6-d64-lr0.001'
    'MemPAE-ws-pos_query+token-d64-lr0.001-t0.05'
    'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1'
    'PAE-ws-d64-lr0.001'
    'MemPAE-ws-d64-lr0.001'
    'MemPAE-ws-pos_query-d64-lr0.001-t0.1'
    'MemPAE-ws-pos_qu2ery-d64-lr0.001-t0.05'
    'MemPAE-ws-pos_query-d64-lr0.001'
)

for model_type in "${model_list[@]}"; do
    echo "Deleteing anomalies in $model_type ";
    rm -r ~/ad/Perceiver-AD/results/$model_type/*anomalies*
    rm -r ~/ad/Perceiver-AD/results/$model_type/*irrelevant_features*
done