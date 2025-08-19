import torch
import numpy as np
import argparse
import os
import json
from utils import build_trainer, replace_config, load_configs, get_input_dim
from scipy import io
import importlib
import glob
import time
import itertools
import copy

npz_files = glob.glob(os.path.join('./Data', '*.npz'))
npz_datanames = [os.path.splitext(os.path.basename(file))[0] for file in npz_files]

mat_files = glob.glob(os.path.join('./Data', '*.mat'))
mat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in mat_files]

DEEP_MODELS = ['Perceiver', 'MCM', 'DRL']
CLASSIC_MODELS = ['OCSVM', 'KNN', 'IForest', 'LOF', 'PCA']


def train_test(args, run, model_config):
    cfg = copy.deepcopy(model_config)
    cfg['run'] = run
    cfg['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg['model_type'] = args.model_type
    cfg['dataset_name'] = args.dataname
    print(cfg)

    Trainer = build_trainer(args)
    trainer = Trainer(model_config=cfg, base_path=args.base_path)
    trainer.training(cfg['epochs'])
    mse_rauc, mse_ap, mse_f1 = trainer.evaluate()
    num_train_samples, num_test_samples = trainer.get_num_instances()

    print(f"[run {run}] AUC-ROC: {mse_rauc:.4f} | AUC-PR: {mse_ap:.4f} | F1: {mse_f1:.4f}")

    cfg.pop("device")
    results_dict = {
        'run': run,
        'AUC-ROC': float(mse_rauc),
        'AUC-PR': float(mse_ap),
        'f1': float(mse_f1),
        'num_train_samples': num_train_samples,
        'num_test_samples': num_test_samples,
        'config': cfg,
    }
    return results_dict


def tune(args):
    os.makedirs(args.base_path, exist_ok=True)
    summary_path = os.path.join(args.base_path, 'tune.json')

    # Load config
    model_config = load_configs(args)
    model_config['train_ratio'] = args.train_ratio
    model_config['data_dim'] = get_input_dim(args, model_config)

    # get hyperparameters for grid search (value is list)
    grid_params = {k: v for k, v in model_config.items() if isinstance(v, list)}
    fixed_params = {k: v for k, v in model_config.items() if not isinstance(v, list)}

    start = time.time()
    keys, values = zip(*grid_params.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    tune_results = []
    best_score = -np.inf
    best_result = None

    for suggest in combinations:
        cfg = {**fixed_params, **suggest}
        result = train_test(args, run=0, model_config=cfg)
        tune_results.append(result)
        if result['AUC-PR'] > best_score:
            best_score = result['AUC-PR']
            best_result = result

    end = time.time()
    summary = {
        'model_config': {
            'model_type': args.model_type,
            'dataset_name': args.dataname,
            'train_ratio': args.train_ratio
        },
        'best_result': best_result,
        'tune_results': tune_results,
        'total_time': end - start
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print("\nSummary")
    print(json.dumps(summary, indent=4))

    return best_result['config']

def main(args):
    os.makedirs(args.base_path, exist_ok=True)
    summary_path = os.path.join(args.base_path, 'summary.json')
    if os.path.exists(summary_path):
        print(f"summary.json already exists at {summary_path}. Skipping execution.")
        return

    # find best hyperparameters
    best_config = tune(args)

    # repeat 10 times with best hyperparameters 
    args.model_config = best_config
    start = time.time()    
    all_results = []
    for i in range(10):
        result = train_test(args, i, best_config)
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
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print("\nSummary")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='Hepatitis')
    parser.add_argument('--model_type', type=str, default='DRL')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--train_ratio', type=float, default=1.0)

    # Perceiver Experiment 
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--mlp_ratio', type=float, default=None)
    parser.add_argument('--dropout_prob', type=float, default=None)
    parser.add_argument('--drop_col_prob', type=float, default=None)

    args = parser.parse_args()
    if args.exp_name is None:
        args.exp_name = args.model_type
    args.base_path = f"results/{args.exp_name}/{args.dataname}/{args.train_ratio}"
    print(args)
    main(args)
