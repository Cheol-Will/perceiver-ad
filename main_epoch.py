import torch
import numpy as np
import argparse
import os
import json
import time
from utils import get_parser, get_logger, build_trainer, load_yaml

def train_test_per_epoch(model_config, train_config, run):
    train_config['run'] = run
    train_config['logger'].info(f"[run {run}]" + '-'*60)
    trainer = build_trainer(model_config, train_config)   
    metrics = trainer.train_test_per_epoch(test_per_epochs=50)
    
    results_dict = {
        'run': run,
        'AUC-ROC': [float(rauc) for rauc in metrics['rauc']],
        'AUC-PR': [float(ap) for ap in metrics['ap']],
        'f1': [float(f1) for f1 in metrics['f1']],
    }
    return results_dict

def main(args):
    os.makedirs(args.base_path, exist_ok=True)

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
        result = train_test_per_epoch(model_config, train_config, seed)
        all_results.append(result)
    end = time.time()
    total_time = end - start
    
    train_config.pop('device')
    train_config.pop('logger')
    
    full_epochs = train_config['epochs']
    epochs = list(range(50, full_epochs+1, 50))
    for idx, epoch in enumerate(epochs):
        if args.contamination_ratio is not None:
            epoch_path = f"results/{args.exp_name}-{epoch}/{args.dataname}_contam{args.contamination_ratio}/{args.train_ratio}"
        else:
            epoch_path = f"results/{args.exp_name}-{epoch}/{args.dataname}/{args.train_ratio}"
        
        os.makedirs(epoch_path, exist_ok=True)
        
        # For each epoch
        epoch_results = []
        for r in all_results:
            epoch_results.append({
                'run': r['run'],
                'AUC-ROC': r['AUC-ROC'][idx],
                'AUC-PR': r['AUC-PR'][idx],
                'f1': r['f1'][idx]
            })
        
        # avg metrics
        mean_metrics = {
            'AUC-ROC': float(np.mean([r['AUC-ROC'][idx] for r in all_results])),
            'AUC-PR': float(np.mean([r['AUC-PR'][idx] for r in all_results])),
            'f1': float(np.mean([r['f1'][idx] for r in all_results]))
        }
        
        # summary 
        summary = {
            'model_config': model_config,
            'train_config': train_config,
            'epoch': epoch,
            'mean_metrics': mean_metrics,
            'total_time': total_time,
            'all_seeds': epoch_results,
        }
        
        summary_path = os.path.join(epoch_path, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\nSummary for epoch {epoch}")
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