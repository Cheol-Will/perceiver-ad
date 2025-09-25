import torch
import numpy as np
import argparse
import os
import json
import time
from utils import get_parser, get_logger, load_yaml

BASELINE_MODELS = ['OCSVM', 'KNN', 'IForest', 'LOF', 'PCA', 'ECOD', 
                   'DeepSVDD', 'GOAD', 'ICL', 'NeuTraL']

def build_analyzer(model_config, train_config, analysis_config):
    model_type = train_config['model_type']
    if model_type == 'Perceiver':
        from models.Perceiver.Analyzer import Analyzer
    elif model_type == 'MemAE':
        from models.MemAE.Analyzer import Analyzer
    elif model_type == 'MemPAE':
        from models.MemPAE.Analyzer import Analyzer
    elif model_type == 'MCM':
        from models.MCM.Analyzer import Analyzer
    elif model_type == 'Disent':
        from models.Disent.Analyzer import Analyzer
    elif model_type == 'PAE':
        from models.PAE.Analyzer import Analyzer        
    elif model_type == 'AutoEncoder':
        from models.AutoEncoder.Analyzer import Analyzer        
    elif model_type in BASELINE_MODELS:
        from models.Baselines.Analyzer import Analyzer
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return Analyzer(model_config, train_config, analysis_config)


def train_test(args, model_config, train_config, analysis_config, run):
    train_config['run'] = run
    train_config['logger'].info(f"[run {run}]" + '-'*60)
    analyzer = build_analyzer(model_config, train_config, analysis_config)    
    
    if analysis_config['compare_regresssion_with_sup_attn']:
        analyzer.training_supervised()
        
    else:
        analyzer.training()

    if analysis_config['plot_recon']:
        analyzer.plot_reconstruction()
    if analysis_config['plot_histogram']:
        analyzer.plot_anomaly_histograms(remove_outliers=True)
    if analysis_config['plot_memory_weight']:
        analyzer.plot_memory_weight()
    if analysis_config['compare_regresssion_with_attn']:
        analyzer.compare_regresssion_with_attn(use_sup_attn=False, lambda_attn=args.lambda_attn)
    if analysis_config['compare_regresssion_with_sup_attn']:
        analyzer.compare_regresssion_with_attn(use_sup_attn=True, lambda_attn=args.lambda_attn, model_type='decision_tree')
    if analysis_config['plot_attn_and_corr']:
        analyzer.plot_attn_and_corr()
    if args.plot_pos_encoding:
        analyzer.plot_pos_encoding(use_mask=True)    
    if args.plot_attn:
        analyzer.plot_attn(use_mask=True)    
    if args.plot_attn_single:
        analyzer.plot_attn_single()
    if args.plot_attn_simple:
        analyzer.plot_attn_simple()
    if args.plot_attn_all_heads:
        analyzer.plot_attn_all_heads()
    if args.plot_attn_all_depths:
        analyzer.plot_attn_all_depths()
    if args.plot_attn_everything:
        analyzer.plot_attn_everything()
    if args.plot_feature_reconstruction_distribution: 
        for i in range(36):
            analyzer.plot_feature_reconstruction_distribution(
                feature_idx=i
            )
    if args.plot_2x2:
        # analyzer.plot_2x4(abnormal_avg=True, plot_heads=False, plot_2x2=True)
        analyzer.plot_2x4(5, plot_heads=False, plot_2x2=True) # wine
        analyzer.plot_2x4(1, plot_heads=False, plot_2x2=True) # pima
    if args.plot_2x4:
        analyzer.analyze_best_abnormal_sample()
        # analyzer.plot_2x4(abnormal_avg=True, plot_heads=False, plot_2x2=True)
        # for i in range(100):
        #     analyzer.plot_2x4(i, plot_heads=False, plot_2x2=True) 
    if args.plot_2x3:
        # to show that transformer are handling input-depedent information.
        analyzer.plot_2x3([0, 1]) 
        # analyzer.plot_2x3([])
    if args.plot_grad_z_x:
        analyzer.plot_grad_z_x()
        
    if args.visualize_attention_vs_shap:
        # analyzer.visualize_attention_vs_shap(save_plots=True)
        analyzer.visualize_attention_vs_shap()
    if args.compare_shap_vs_encoder_attention_per_sample:
        analyzer.compare_shap_vs_encoder_attention_per_sample(plot_group_average=True, include_self_attention=True)
        analyzer.compare_shap_vs_encoder_attention_per_sample(plot_group_average=True, include_self_attention=False)

    if args.plot_tsne_memory_addressing:
        analyzer.plot_tsne_memory_addressing()

    return 



def main(args):
    os.makedirs(args.base_path, exist_ok=True)
    summary_path = os.path.join(args.base_path, 'summary.json')
    if os.path.exists(summary_path):
        print(f"summary.json already exists at {summary_path}. Skipping execution.")
        return

    logger = get_logger(os.path.join(args.base_path, 'log.log'))

    model_config, train_config = load_yaml(args)
    train_config['logger'] = logger
    train_config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_config['base_path'] = args.base_path
    if train_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn', force=True)

    analysis_config = {}
    analysis_config['plot_attn'] = args.plot_attn
    analysis_config['plot_recon'] = args.plot_recon
    analysis_config['plot_histogram'] = args.plot_histogram
    analysis_config['plot_memory_weight'] = args.plot_memory_weight
    analysis_config['compare_regresssion_with_attn'] = args.compare_regresssion_with_attn
    analysis_config['compare_regresssion_with_sup_attn'] = args.compare_regresssion_with_sup_attn
    analysis_config['plot_attn_and_corr'] = args.plot_attn_and_corr
    analysis_config['plot_tsne_recon'] = args.plot_tsne_recon
    
    
    start = time.time()    
    for seed in range(1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        train_test(args, model_config, train_config, analysis_config, seed)

    end = time.time()
    total_time = end - start
    summary = {
        'model_config': {
            'model_type': args.model_type,
            'dataset_name': args.dataname,
            'train_ratio': args.train_ratio
        },
    }
    # with open(summary_path, 'w') as f:
    #     json.dump(summary, f, indent=4)
    print("\nSummary")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    parser = get_parser()

    # Analysis arguments
    parser.add_argument('--plot_attn', action='store_true')
    parser.add_argument('--plot_recon', action='store_true')
    parser.add_argument('--plot_histogram', action='store_true')
    parser.add_argument('--plot_memory_weight', action='store_true')
    parser.add_argument('--compare_regresssion_with_attn', action='store_true')
    parser.add_argument('--lambda_attn', type=float, default=1.0)
    parser.add_argument('--compare_regresssion_with_sup_attn', action='store_true')
    parser.add_argument('--plot_attn_and_corr', action='store_true')
    parser.add_argument('--plot_tsne_recon', action='store_true')
    parser.add_argument('--plot_pos_encoding', action='store_true')
    parser.add_argument('--plot_tsne_latent_vs_memory', action='store_true')
    parser.add_argument('--use_latents_avg', action='store_true')
    parser.add_argument('--plot_tsne_memory_separate', action='store_true')
    parser.add_argument('--plot_tsne_original_with_memory', action='store_true')
    parser.add_argument('--plot_umap_latent_vs_memory', action='store_true')
    # parser.add_argument('--plot_umap_latent_vs_memory', action='store_true')
    parser.add_argument('--plot_attn_single', action='store_true')
    parser.add_argument('--plot_hist_diff_memory_addressing', action='store_true')
    parser.add_argument('--get_single_samples', action='store_true')
    parser.add_argument('--plot_attn_simple', action='store_true')
    parser.add_argument('--plot_attn_all_heads', action='store_true')
    parser.add_argument('--plot_attn_all_depths', action='store_true')
    parser.add_argument('--plot_attn_everything', action='store_true')
    parser.add_argument('--plot_attn_single_self_sum', action='store_true')
    parser.add_argument('--plot_attn_dec_memory', action='store_true')
    parser.add_argument('--plot_attn_pair', action='store_true')
    parser.add_argument('--plot_feature_reconstruction_distribution', action='store_true')
    parser.add_argument('--plot_2x4', action='store_true')
    parser.add_argument('--plot_2x3', action='store_true')
    parser.add_argument('--plot_2x2', action='store_true')
    parser.add_argument('--plot_grad_z_x', action='store_true')
    parser.add_argument('--visualize_attention_vs_shap', action='store_true')
    parser.add_argument('--compare_shap_vs_encoder_attention_per_sample', action='store_true')
    parser.add_argument('--plot_tsne_memory_addressing', action='store_true')

    args = parser.parse_args()
    if args.exp_name is None:
        args.exp_name = args.model_type 
    args.base_path = f"results_analysis/{args.exp_name}/{args.dataname}/{args.train_ratio}"
    main(args)