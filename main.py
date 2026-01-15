import torch
import numpy as np
import argparse
import os
import json
import time
from torch.utils.tensorboard import SummaryWriter
from utils import get_parser, get_logger, build_trainer, load_yaml

def train_test_npt(model_config, train_config, run):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    train_config['run'] = run
    
    if local_rank == 0:
        train_config['logger'].info(f"[run {run}]" + '-'*60)

    trainer = build_trainer(model_config, train_config)   
    start_train = time.time()    
    epochs = trainer.training()
    end_train = time.time()
    train_time = end_train - start_train
    
    start_test = time.time()    
    metrics = trainer.evaluate()
    end_test = time.time()    
    test_time = end_test - start_test
    
    if local_rank == 0:
        if metrics is not None:
            log_str = f"[run {run}]"
            for key, value in metrics.items():
                if isinstance(value, float):
                    log_str += f" {key}: {value:.4f} |"
            train_config['logger'].info(log_str.rstrip(' |'))
    
            results_dict = {
                'run': run,
                'train_time': train_time,
                'test_time': test_time,
                'epochs': epochs,
            }
            for key, value in metrics.items():
                results_dict[key] = float(value) if isinstance(value, (int, float, np.floating)) else value
            
            return results_dict    
    return None


def train_test(model_config, train_config, run):
    train_config['run'] = run
    train_config['logger'].info(f"[run {run}]" + '-'*60)
    trainer = build_trainer(model_config, train_config)   
    start_train = time.time()    
    epochs = trainer.training()
    end_train = time.time()
    train_time = end_train - start_train
    
    start_test = time.time()    
    metrics = trainer.evaluate()
    end_test = time.time()    
    test_time = end_test - start_test
    
    log_str = f"[run {run}]"
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)):
            log_str += f" {key}: {value:.4f} |"
    train_config['logger'].info(log_str.rstrip(' |'))
    
    results_dict = {
        'run': run,
        'train_time': train_time,
        'test_time': test_time,
        'epochs': epochs,
    }

    # add every metric    
    for key, value in metrics.items():
        results_dict[key] = float(value) if isinstance(value, (int, float, np.floating)) else value
    
    return results_dict


def main(args):
    is_nptad = args.model_type == 'NPTAD'
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_nptad else 0
    
    os.makedirs(args.base_path, exist_ok=True)
    summary_path = os.path.join(args.base_path, 'summary.json')
    
    all_results = []
    if local_rank == 0 and os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            existing_summary = json.load(f)
            all_results = existing_summary.get('all_seeds', [])
        print(f"Resumed from {summary_path}. Loaded {len(all_results)} runs.")

    if len(all_results) >= args.runs:
        if local_rank == 0:
            print(f"All runs already exist. Skipping.")
    
    logger = get_logger(os.path.join(args.base_path, 'log.log'))
    model_config, train_config = load_yaml(args, parser)

    train_config['logger'] = logger
    train_config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    writer = None
    if local_rank == 0:
        if args.contamination_ratio is not None:
            tensorboard_dir = f"results/{args.exp_name}/tensorboard_contam{args.contamination_ratio}/{args.train_ratio}"
        else:
            tensorboard_dir = f"results/{args.exp_name}/tensorboard/{args.train_ratio}"
        
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, args.dataname))
        print(f"TensorBoard logs will be saved to: {tensorboard_dir}/{args.dataname}")
    
    train_config['writer'] = writer
    
    if train_config['num_workers'] > 0:
        torch.multiprocessing.set_start_method('spawn', force=True)
    
    if local_rank == 0:
        print(model_config)
        print(train_config)
    
    start = time.time()    
    
    for seed in range(args.runs):
        if any(r['run'] == seed for r in all_results):
            if local_rank == 0:
                print(f"[run {seed}] Already exists. Skipping...")
            continue

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        current_train_config = train_config.copy()

        if train_config["model_type"] != 'NPTAD':
            result = train_test(model_config, current_train_config, seed) 
            if result is not None:
                all_results.append(result)
        else:
            result = train_test_npt(model_config, current_train_config, seed)
            if local_rank == 0 and result is not None:
                all_results.append(result)
        
        if local_rank == 0 and len(all_results) > 0:
            # Extract metric keys
            exclude_keys = {'run', 'train_time', 'test_time', 'epochs'}
            metric_keys = []
            if all_results:
                # Union of all keys to be safe
                all_keys = set().union(*(d.keys() for d in all_results))
                metric_keys = [k for k in all_keys if k not in exclude_keys]
            
            # Aggregation
            mean_metrics = {}
            std_metrics = {}
            for key in metric_keys:
                values = [r[key] for r in all_results if key in r and isinstance(r[key], (int, float, np.floating))]
                if values:
                    mean_metrics[key] = float(np.mean(values))
                    std_metrics[f"{key}_std"] = float(np.std(values))
            
            save_train_config = train_config.copy()
            save_train_config.pop('device', None)
            save_train_config.pop('logger', None)
            save_train_config.pop('writer', None)
            
            total_time = time.time() - start # Approximate total time
            
            summary = {
                'model_config': model_config,
                'train_config': save_train_config,
                'mean_metrics': mean_metrics,
                'std_metrics': std_metrics,
                'total_time': total_time,
                'all_seeds': all_results,
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            
            print(f"Saved results for run {seed} to {summary_path}")

    if local_rank == 0 and len(all_results) > 0:
        print(f"\n{'='*80}")
        print("Final Summary")
        print(f"{'='*80}")
    
        for key in sorted(mean_metrics.keys()):
            std_key = f"{key}_std"
            if std_key in std_metrics:
                print(f"  {key}: {mean_metrics[key]:.4f} ± {std_metrics[std_key]:.4f}")
            else:
                print(f"  {key}: {mean_metrics[key]:.4f}")
        print(f"{'='*80}\n")
        
    if writer is not None:
        writer.close()

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