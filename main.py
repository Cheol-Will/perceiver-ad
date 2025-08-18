import torch
import numpy as np
import argparse
import os
import json
from utils import build_trainer, replace_config
from scipy import io
import importlib
import glob
import time

npz_files = glob.glob(os.path.join('./Data', '*.npz'))
npz_datanames = [os.path.splitext(os.path.basename(file))[0] for file in npz_files]

mat_files = glob.glob(os.path.join('./Data', '*.mat'))
mat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in mat_files]


# def 


def train_test(args, seed):
    # Load config
    dict_to_import = 'model_config_' + args.model_type
    module_name = 'configs'
    module = importlib.import_module(module_name)
    model_config = getattr(module, dict_to_import)

    model_config['random_seed'] = seed
    model_config['train_ratio'] = args.train_ratio

    if args.model_type == 'Perceiver':
        model_config = replace_config(args, model_config)

    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    if model_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn', force=True)

    # Extract input dimension
    if args.dataname in npz_datanames:
        path = os.path.join(model_config['data_dir'], args.dataname + '.npz')
        data = np.load(path)
    elif args.dataname in mat_datanames:
        path = os.path.join(model_config['data_dir'], args.dataname + '.mat')
        data = io.loadmat(path)
    else:
        raise ValueError(f"Unknown dataset {args.dataname}")

    samples = data['X']
    model_config['seed'] = seed
    model_config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config['model_type'] = args.model_type
    model_config['dataset_name'] = args.dataname
    model_config['data_dim'] = samples.shape[-1]
    print(model_config)
    # train and test
    Trainer = build_trainer(args)
    trainer = Trainer(model_config=model_config, base_path=args.base_path)
    trainer.training(model_config['epochs'])
    mse_rauc, mse_ap, mse_f1 = trainer.evaluate()
    num_train_samples, num_test_samples = trainer.get_num_instances()
    
    print(f"[Seed {seed}] AUC-ROC: {mse_rauc:.4f} | AUC-PR: {mse_ap:.4f} | F1: {mse_f1:.4f}")

    results_dict = {
        'seed': seed,
        'AUC-ROC': float(mse_rauc),
        'AUC-PR': float(mse_ap),
        'f1': float(mse_f1),
        'num_train_samples': num_train_samples,
        'num_test_samples': num_test_samples,
    }

    return results_dict


def main(args):
    os.makedirs(args.base_path, exist_ok=True)
    summary_path = os.path.join(args.base_path, 'summary.json')
    if os.path.exists(summary_path):
        print(f"summary.json already exists at {summary_path}. Skipping execution.")
        return
    start = time.time()    
    all_results = []
    for seed in range(10):
        result = train_test(args, seed)
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
    main(args)