import torch
import numpy as np
import argparse
import os
import json
import time
from utils import get_logger, load_yaml

BASELINE_MODELS = ['OCSVM', 'KNN', 'IForest', 'LOF', 'PCA', 'ECOD', 
                   'DeepSVDD', 'AutoEncoder', 'GOAD', 'ICL', 'NeuTraL']

def build_analyzer(model_config, train_config, analysis_config):
    model_type = train_config['model_type']
    if model_type == 'Perceiver':
        from models.Perceiver.Analyzer import Analyzer
    elif model_type == 'MemAE':
        from models.MemAE.Analyzer import Analyzer
    elif model_type == 'MemPAE':
        from models.MemPAE.Analyzer import Analyzer
    elif model_type == 'PAE':
        from models.PAE.Analyzer import Analyzer        
    elif model_type in BASELINE_MODELS:
        from models.Baselines.Analyzer import Analyzer
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return Analyzer(model_config, train_config, analysis_config)


def train_test(model_config, train_config, analysis_config, run):
    train_config['run'] = run
    train_config['logger'].info(f"[run {run}]" + '-'*60)
    analyzer = build_analyzer(model_config, train_config, analysis_config)    
    analyzer.training()
    
    if analysis_config['plot_recon']:
        analyzer.plot_reconstruction()
    if analysis_config['plot_histogram']:
        analyzer.plot_anomaly_histograms(remove_outliers=True)
    if analysis_config['plot_memory_weight']:
        analyzer.plot_memory_weight()
    if analysis_config['compare_regresssion_with_attn']:
        analyzer.compare_regresssion_with_attn()
    if analysis_config['plot_tsne_recon']:
        # analyzer.plot_tsne_reconstruction()
        analyzer.plot_combined_tsne()
        
    
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
    analysis_config['plot_tsne_recon'] = args.plot_tsne_recon
    
    
    start = time.time()    
    for seed in range(1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        train_test(model_config, train_config, analysis_config, seed)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='Hepatitis')
    parser.add_argument('--model_type', type=str, default='DRL')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--train_ratio', type=float, default=1.0)

    # Experiment 
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)

    # Experiment
    parser.add_argument('--mlp_ratio', type=float, default=None)
    parser.add_argument('--dropout_prob', type=float, default=None)
    parser.add_argument('--drop_col_prob', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--sim_type', type=str, default=None)
    parser.add_argument('--num_repeat', type=int, default=None)
    parser.add_argument('--is_weight_sharing', action='store_true')
    parser.add_argument('--use_pos_enc_as_query', action='store_true')
    parser.add_argument('--num_latents', type=int, default=None)
    parser.add_argument('--num_adapters', type=int, default=None)
    parser.add_argument('--use_vq_loss_as_score', action='store_true')    
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--shrink_thred', type=float, default=None)
    parser.add_argument('--latent_loss_weight', type=float, default=None)
    parser.add_argument('--entropy_loss_weight', type=float, default=None)
    parser.add_argument('--use_entropy_loss_as_score', action='store_true')    
    parser.add_argument('--use_mask_token', action='store_true')    
    parser.add_argument('--not_use_power_of_two', action='store_true')    
    parser.add_argument('--num_memories_not_use_power_of_two', action='store_true')    
    parser.add_argument('--num_memories_twice', action='store_true')    
    parser.add_argument('--is_recurrent', action='store_true')    
    parser.add_argument('--contamination_ratio', type=float, default=None)    
    parser.add_argument('--top_k', type=int, default=None)    

    # Analysis arguments
    parser.add_argument('--plot_attn', action='store_true')
    parser.add_argument('--plot_recon', action='store_true')
    parser.add_argument('--plot_histogram', action='store_true')
    parser.add_argument('--plot_memory_weight', action='store_true')
    parser.add_argument('--compare_regresssion_with_attn', action='store_true')
    parser.add_argument('--plot_tsne_recon', action='store_true')

    args = parser.parse_args()
    if args.exp_name is None:
        args.exp_name = args.model_type 
    args.base_path = f"results_analysis/{args.exp_name}/{args.dataname}/{args.train_ratio}"
    main(args)