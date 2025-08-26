import torch
import numpy as np
import argparse
import os
import json
import time
from utils import get_logger, load_yaml
from models.Perceiver.Trainer import Trainer
from models.Perceiver.Analyzer import Analyzer

def train_test(model_config, run):
    model_config['run'] = run
    model_config['logger'].info(f"[run {run}]" + '-'*60)
    trainer = Analyzer(model_config)    
    trainer.training()
    mse_rauc, mse_ap, mse_f1 = trainer.evaluate()
    model_config['logger'].info(f"[run {run}] AUC-ROC: {mse_rauc:.4f} | AUC-PR: {mse_ap:.4f} | F1: {mse_f1:.4f}")
    results_dict = {
        'run': run,
        'AUC-ROC': float(mse_rauc),
        'AUC-PR': float(mse_ap),
        'f1': float(mse_f1),
    }

    if model_config['plot_recon']:
        trainer.plot_reconstruction()
    if model_config['plot_histogram']:
        trainer.plot_anomaly_histograms(remove_outliers=True)
        
    return results_dict


def main(args):
    os.makedirs(args.base_path, exist_ok=True)
    summary_path = os.path.join(args.base_path, 'summary.json')
    if os.path.exists(summary_path):
        print(f"summary.json already exists at {summary_path}. Skipping execution.")
        return

    logger = get_logger(os.path.join(args.base_path, 'log.log'))
    model_config = load_yaml(args)
    model_config['logger'] = logger
    model_config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn', force=True)

    model_config['plot_attn'] = args.plot_attn
    model_config['plot_recon'] = args.plot_recon
    model_config['plot_histogram'] = args.plot_histogram
    
    start = time.time()    
    all_results = []
    for seed in range(1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        result = train_test(model_config, seed)
        all_results.append(result)
    end = time.time()
    total_time = end - start
    mean_metrics = {
        'AUC-ROC': float(np.mean([r['AUC-ROC'] for r in all_results])),
        'AUC-PR': float(np.mean([r['AUC-PR'] for r in all_results])),
        'f1': float(np.mean([r['f1'] for r in all_results]))
    }
    summary = {
        'model_config': {
            'model_type': args.model_type,
            'dataset_name': args.dataname,
            'train_ratio': args.train_ratio
        },
        'mean_metrics': mean_metrics,
        'total_time': total_time,
        'all_seeds': all_results,
    }
    # with open(summary_path, 'w') as f:
    #     json.dump(summary, f, indent=4)
    print("\nSummary")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='Hepatitis')
    parser.add_argument('--model_type', type=str, default='Perceiver')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--train_ratio', type=float, default=1.0)

    # Perceiver Experiment 
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--mlp_ratio', type=float, default=None)
    parser.add_argument('--dropout_prob', type=float, default=None)
    parser.add_argument('--drop_col_prob', type=float, default=None)
    
    # Analysis arguments
    parser.add_argument('--plot_attn', action='store_true')
    parser.add_argument('--plot_recon', action='store_true')
    parser.add_argument('--plot_histogram', action='store_true')


    args = parser.parse_args()
    if args.exp_name is None:
        args.exp_name = args.model_type 
    args.base_path = f"results_analysis/{args.exp_name}/{args.dataname}/{args.train_ratio}"
    main(args)
