# main.py
import argparse
import torch
import numpy as np
import os
import json
import time
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from models.LATTEMultitask.Trainer import Trainer
from utils import load_yaml_multitask

def aggregate_metrics_across_seeds(all_seed_results):
    """Aggregate metrics across all seeds for each epoch"""
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Collect all metrics: seed -> epoch -> dataset -> metric -> value
    for seed, epoch_metrics in all_seed_results.items():
        for epoch, dataset_metrics in epoch_metrics.items():
            for dataset_name, metrics in dataset_metrics.items():
                for metric_name, value in metrics.items():
                    aggregated[epoch][dataset_name][metric_name].append(value)
    
    # Calculate mean and std
    results = {}
    for epoch in sorted(aggregated.keys()):
        results[epoch] = {}
        for dataset_name in sorted(aggregated[epoch].keys()):
            results[epoch][dataset_name] = {}
            for metric_name, values in aggregated[epoch][dataset_name].items():
                results[epoch][dataset_name][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'values': values
                }
    
    return results

def compute_average_metrics(aggregated_results, target_metric='ap'):
    """Compute average metric across all datasets for each epoch"""
    epoch_averages = {}
    
    for epoch, dataset_metrics in aggregated_results.items():
        values = []
        for dataset_name, metrics in dataset_metrics.items():
            if target_metric in metrics:
                values.append(metrics[target_metric]['mean'])
        
        if values:
            epoch_averages[epoch] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
    
    return epoch_averages

def train_test_multitask(model_config, train_config, seed):
    """Train and test for a single seed"""
    train_config['run'] = seed
    trainer = Trainer(model_config, train_config)
    epoch_metrics = trainer.train_test()
    return epoch_metrics

def main(args):
    os.makedirs(args.base_path, exist_ok=True)
    summary_path = os.path.join(args.base_path, 'summary.json')
    
    # Load configurations
    model_config, train_config = load_yaml_multitask(args)
    train_config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_config['base_path'] = args.base_path  # ← 추가: checkpoint 저장 경로 설정
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.base_path, 'tensorboard'))
    train_config['writer'] = writer
    
    print(f"Starting multi-seed training with {args.runs} seeds")
    print(f"Results will be saved to: {args.base_path}")
    print(f"Checkpoints will be saved to: {os.path.join(args.base_path, 'checkpoints')}")
    print("="*100)
    
    start = time.time()
    all_seed_results = {}
    
    # Train across multiple seeds
    for seed in range(args.runs):
        print(f"\n{'#'*100}")
        print(f"# Running seed {seed + 1}/{args.runs}")
        print(f"{'#'*100}\n")
        
        # Set seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        # Train and collect results
        epoch_metrics = train_test_multitask(model_config, train_config, seed)
        all_seed_results[seed] = epoch_metrics
    
    end = time.time()
    total_time = end - start
    
    # Aggregate results across seeds
    print("\n" + "="*100)
    print("Aggregating results across all seeds...")
    print("="*100)
    
    aggregated_results = aggregate_metrics_across_seeds(all_seed_results)
    
    # Compute final average metrics (using last epoch)
    final_epoch = max(aggregated_results.keys())
    final_metrics = {}
    
    for dataset_name, metrics in aggregated_results[final_epoch].items():
        final_metrics[dataset_name] = {
            'rauc': metrics['rauc']['mean'],
            'ap': metrics['ap']['mean'],
            'f1': metrics['f1']['mean']
        }
    
    # Overall average across all datasets
    overall_metrics = {
        'rauc': np.mean([m['rauc']['mean'] for m in aggregated_results[final_epoch].values()]),
        'ap': np.mean([m['ap']['mean'] for m in aggregated_results[final_epoch].values()]),
        'f1': np.mean([m['f1']['mean'] for m in aggregated_results[final_epoch].values()])
    }
    
    # Prepare summary
    train_config_serializable = {k: v for k, v in train_config.items() 
                                  if k not in ['device', 'writer']}
    train_config_serializable['device'] = str(train_config['device'])
    
    summary = {
        'model_config': model_config,
        'train_config': train_config_serializable,
        'num_seeds': args.runs,
        'total_time': total_time,
        'final_epoch': final_epoch,
        'overall_metrics': overall_metrics,
        'per_dataset_metrics': final_metrics,
        'aggregated_results': {str(k): v for k, v in aggregated_results.items()},
        'individual_seeds': {str(k): {str(e): m for e, m in v.items()} 
                            for k, v in all_seed_results.items()}
    }
    
    # Save summary
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save readable summary
    readable_path = os.path.join(args.base_path, 'summary_readable.txt')
    with open(readable_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("MULTITASK TRAINING SUMMARY\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Number of seeds: {args.runs}\n")
        f.write(f"Total training time: {total_time/60:.2f} minutes\n")
        f.write(f"Final epoch: {final_epoch}\n\n")
        
        f.write("Overall Metrics (averaged across all datasets):\n")
        f.write("-"*100 + "\n")
        f.write(f"  RAUC: {overall_metrics['rauc']:.4f}\n")
        f.write(f"  AP:   {overall_metrics['ap']:.4f}\n")
        f.write(f"  F1:   {overall_metrics['f1']:.4f}\n\n")
        
        f.write("Per-Dataset Metrics at Final Epoch:\n")
        f.write("-"*100 + "\n")
        for dataset_name, metrics in final_metrics.items():
            f.write(f"\n{dataset_name}:\n")
            dataset_agg = aggregated_results[final_epoch][dataset_name]
            f.write(f"  RAUC: {metrics['rauc']:.4f} ± {dataset_agg['rauc']['std']:.4f}\n")
            f.write(f"  AP:   {metrics['ap']:.4f} ± {dataset_agg['ap']['std']:.4f}\n")
            f.write(f"  F1:   {metrics['f1']:.4f} ± {dataset_agg['f1']['std']:.4f}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("Epoch-wise Average AP (across all datasets):\n")
        f.write("="*100 + "\n")
        epoch_ap = compute_average_metrics(aggregated_results, 'ap')
        for epoch in sorted(epoch_ap.keys()):
            f.write(f"Epoch {epoch}: {epoch_ap[epoch]['mean']:.4f} ± {epoch_ap[epoch]['std']:.4f}\n")
    
    # Print summary
    print("\n" + "="*100)
    print("TRAINING COMPLETE")
    print("="*100)
    print(f"\nTotal time: {total_time/60:.2f} minutes")
    print(f"\nOverall metrics (averaged across {len(final_metrics)} datasets):")
    print(f"  RAUC: {overall_metrics['rauc']:.4f}")
    print(f"  AP:   {overall_metrics['ap']:.4f}")
    print(f"  F1:   {overall_metrics['f1']:.4f}")
    print(f"\nResults saved to:")
    print(f"  JSON: {summary_path}")
    print(f"  Readable: {readable_path}")
    print(f"  TensorBoard: {os.path.join(args.base_path, 'tensorboard')}")
    print(f"  Checkpoints: {os.path.join(args.base_path, 'checkpoints')}")
    
    writer.close()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='Hepatitis')
    parser.add_argument('--model_type', type=str, default='DRL')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--train_ratio', type=float, default=1.0)

    # train config
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--sche_gamma', type=float, default=None)

    # model config
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--num_latents', type=int, default=None)
    parser.add_argument('--num_memories', type=int, default=None)
    parser.add_argument('--is_weight_sharing', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--sim_type', type=str, default='cos')
    
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.dataname = None  # Multitask
    if args.exp_name is None:
        args.exp_name = args.model_type
    args.base_path = f"results/{args.exp_name}/{args.train_ratio}"
    
    main(args)
