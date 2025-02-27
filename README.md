# DRL: Decomposed Representation Learning for Tabular Anomaly Detection

## Official implementation of the experiments in the [DRL paper](https://openreview.net/forum?id=CJnceDksRd).

## Instructions

First, construct the file "./models" and "./results":

    mkdir models
    mkdir results
    
Second, run the command:

    python main.py --dataname Hepatitis --model_type DRL --preprocess standard --diversity True --plearn False --input_info True --input_info_ratio 0.1 --cl True --cl_ratio 0.06 --basis_vector_num 5 --seed 42


## Citing the paper
If our work is useful for your own, you can cite us with the following BibTex entry:

    @inproceedings{
    ye2025drl,
    title={{DRL}: Decomposed Representation Learning for Tabular Anomaly Detection},
    author={Hangting Ye and He Zhao and Wei Fan and Mingyuan Zhou and Dan dan Guo and Yi Chang},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=CJnceDksRd}
    }
