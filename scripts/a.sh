data_list=(
    # wine 
    # glass 
    # wbc 
    # ionosphere 
    # arrhythmia 
    # breastw 
    # pima  
    # optdigits 

    # cardio 
    # cardiotocography 
    # thyroid 
    # satellite 
    # "satimage-2" 
    # pendigits

    mammography 
    campaign 
    shuttle 

    # fraud 
    # nslkdd 
    # census
)

for data in "${data_list[@]}"; do
    cp -r results/TADAM-tuned--recon_weight0.01_knn1/$data/ results/TADAM-tuned_knn
    # 
    # rm results_analysis/TADAM-default-knn16_/
    # rm results_analysis/TAECL-temp0.2-contra0.01/$data/1.0/model.pt
    # cp -r results/TMLM-d128-lr0.001-mask0.1-r50/$data results/TMLM-tuned/
    # echo "Copying pdf in $data";
    # cp results_analysis/MemPAE-ws-pos_query+token-d64-lr0.001-t0.1/$data/1.0/attention_2x4_comparison_idx0_$data.pdf results_analysis_paper/further_analysis/
done