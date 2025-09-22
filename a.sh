data_list=(
    pima
    wine
    wbc
    thyroid
    pendigits
    shuttle
)

for data in "${data_list[@]}"; do
    echo "Copying pdf in $data";
    cp results_analysis/MemPAE-ws-pos_query+token-d64-lr0.001-t0.1/$data/1.0/attention_2x4_comparison_idx0_$data.pdf results_analysis_paper/further_analysis/
    cp results_analysis/MemPAE-ws-pos_query+token-d64-lr0.001-t0.1/$data/1.0/attention_2x4_comparison_idx1_$data.pdf results_analysis_paper/further_analysis/
    cp results_analysis/MemPAE-ws-pos_query+token-d64-lr0.001-t0.1/$data/1.0/attention_2x4_comparison_idx2_$data.pdf results_analysis_paper/further_analysis/
    cp results_analysis/MemPAE-ws-pos_query+token-d64-lr0.001-t0.1/$data/1.0/attention_2x4_comparison_avg_$data.pdf results_analysis_paper/further_analysis/
done