import torch
import numpy as np
import argparse
import os
import json
import time
from utils import get_parser, get_logger, build_trainer, load_yaml

def train_test(model_config, train_config, run):
    train_config['run'] = run
    train_config['logger'].info(f"[run {run}]" + '-'*60)
    trainer = build_trainer(model_config, train_config)   
    start_train = time.time()    
    trainer.training()
    end_train = time.time()
    train_time = end_train - start_train
    
    start_test = time.time()    
    mse_rauc, mse_ap, mse_f1 = trainer.evaluate()
    end_test = time.time()    
    test_time = end_test - start_test
    
    train_config['logger'].info(f"[run {run}] AUC-ROC: {mse_rauc:.4f} | AUC-PR: {mse_ap:.4f} | F1: {mse_f1:.4f}")
    results_dict = {
        'run': run,
        'AUC-ROC': float(mse_rauc),
        'AUC-PR': float(mse_ap),
        'f1': float(mse_f1),
        'train_time': train_time,
        'test_time': test_time,
    }
    return results_dict

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
    if train_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn', force=True)
    print(model_config)
    print(train_config)
    start = time.time()    
    all_results = []
    for seed in range(args.runs):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        result = train_test(model_config, train_config, seed)
        all_results.append(result)
    end = time.time()
    total_time = end - start
    mean_metrics = {
        'AUC-ROC': float(np.mean([r['AUC-ROC'] for r in all_results])),
        'AUC-PR': float(np.mean([r['AUC-PR'] for r in all_results])),
        'f1': float(np.mean([r['f1'] for r in all_results]))
    }
    train_config.pop('device')
    train_config.pop('logger')
    summary = {
        'model_config': model_config,
        'train_config': train_config,
        'mean_metrics': mean_metrics,
        'total_time': total_time,
        'all_seeds': all_results,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print("\nSummary")
    print(json.dumps(summary, indent=4))

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.exp_name is None:
        args.exp_name = args.model_type 
    if args.contamination_ratio is not None:
        args.base_path = f"results/{args.exp_name}/{args.dataname}_contam{args.contamination_ratio}/{args.train_ratio}"
    else:
        args.base_path = f"results/{args.exp_name}/{args.dataname}/{args.train_ratio}"

    main(args)