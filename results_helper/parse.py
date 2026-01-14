import os
import json
import copy
import argparse

METRIC_SUFFIXES = [
    "rauc", 
    "ap", 
    "f1", 
    "avg_normal", 
    "avg_abnormal",
    "rauc_std", 
    "ap_std", 
    "f1_std", 
    "avg_normal_std", 
    "avg_abnormal_std"
]

COMMON_KEYS = ["run", "epochs", "train_time", "test_time"]

def transform_keys(metrics_dict, prefix):
    new_metrics = {}
    for suffix in METRIC_SUFFIXES:
        target_key = f"{prefix}{suffix}"
        if target_key in metrics_dict:
            new_metrics[suffix] = metrics_dict[target_key]
    return new_metrics

def main(args):
    source_dir = os.path.join(args.results_dir, args.target)
    
    if not os.path.exists(source_dir):
        print(f"Directory not found: {source_dir}")
        return

    modes = ["combined", "contra"]

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file == "summary.json":
                src_path = os.path.join(root, file)
                
                for mode in modes:
                    target_root_dir = f"{source_dir}-{mode}"
                    dest_path = src_path.replace(source_dir, target_root_dir, 1)
                    
                    try:
                        with open(src_path, 'r') as f:
                            data = json.load(f)
                        
                        new_data = copy.deepcopy(data)
                        prefix = f"{mode}_"
                        
                        converted_mean = transform_keys(data.get("mean_metrics", {}), prefix)
                        if converted_mean: 
                            new_data["mean_metrics"] = converted_mean
                        
                        converted_std = transform_keys(data.get("std_metrics", {}), prefix)
                        if converted_std: 
                            new_data["std_metrics"] = converted_std
                        
                        new_seeds = []
                        if "all_seeds" in data:
                            for seed_data in data["all_seeds"]:
                                converted_seed = transform_keys(seed_data, prefix)
                                if converted_seed:
                                    for ck in COMMON_KEYS:
                                        if ck in seed_data: 
                                            converted_seed[ck] = seed_data[ck]
                                    new_seeds.append(converted_seed)
                            new_data["all_seeds"] = new_seeds
                            
                        if "train_config" in new_data:
                            old_exp_name = new_data["train_config"].get("exp_name", args.target)
                            new_data["train_config"]["exp_name"] = f"{old_exp_name}-{mode}"
                            
                            if "base_path" in new_data["train_config"]:
                                new_data["train_config"]["base_path"] = new_data["train_config"]["base_path"].replace(args.target, f"{args.target}-{mode}")

                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        with open(dest_path, 'w') as f:
                            json.dump(new_data, f, indent=4)
                            
                    except Exception as e:
                        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()
    main(args)