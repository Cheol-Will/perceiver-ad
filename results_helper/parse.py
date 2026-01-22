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
    "avg_abnormal_std",
]

COMMON_KEYS = ["run", "epochs", "train_time", "test_time"]


def transform_keys(metrics_dict, prefix):
    new_metrics = {}
    for suffix in METRIC_SUFFIXES:
        target_key = f"{prefix}{suffix}"
        if target_key in metrics_dict:
            new_metrics[suffix] = metrics_dict[target_key]
    return new_metrics


def _extract_modes_from_keys(keys):
    """
    Expect keys like: "{mode}_{suffix}" where suffix is one of METRIC_SUFFIXES.
    Mode can contain underscores freely.
    """
    suffix_patterns = [f"_{s}" for s in METRIC_SUFFIXES]
    modes = set()

    for k in keys:
        if not isinstance(k, str):
            continue
        for pat in suffix_patterns:
            if k.endswith(pat):
                mode = k[: -len(pat)]
                if mode:
                    modes.add(mode)
                break
    return modes


def discover_modes_from_summary(data):
    modes = set()

    mean_metrics = data.get("mean_metrics", {})
    if isinstance(mean_metrics, dict):
        modes |= _extract_modes_from_keys(mean_metrics.keys())

    std_metrics = data.get("std_metrics", {})
    if isinstance(std_metrics, dict):
        modes |= _extract_modes_from_keys(std_metrics.keys())

    all_seeds = data.get("all_seeds", [])
    if isinstance(all_seeds, list):
        for seed_data in all_seeds:
            if isinstance(seed_data, dict):
                modes |= _extract_modes_from_keys(seed_data.keys())

    # deterministic order for later iteration
    return sorted(modes)


def main(args):
    source_dir = os.path.join(args.results_dir, args.target)

    if not os.path.exists(source_dir):
        print(f"Directory not found: {source_dir}")
        return

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file != "summary.json":
                continue

            src_path = os.path.join(root, file)

            try:
                with open(src_path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Failed to read {src_path}: {e}")
                continue

            modes = discover_modes_from_summary(data)
            if not modes:
                print(f"No modes found in: {src_path}")
                continue

            for mode in modes:
                target_root_dir = f"{source_dir}-{mode}"
                dest_path = src_path.replace(source_dir, target_root_dir, 1)

                try:
                    new_data = copy.deepcopy(data)
                    prefix = f"{mode}_"

                    converted_mean = transform_keys(data.get("mean_metrics", {}), prefix)
                    if converted_mean:
                        new_data["mean_metrics"] = converted_mean

                    converted_std = transform_keys(data.get("std_metrics", {}), prefix)
                    if converted_std:
                        new_data["std_metrics"] = converted_std

                    if "all_seeds" in data and isinstance(data["all_seeds"], list):
                        new_seeds = []
                        for seed_data in data["all_seeds"]:
                            if not isinstance(seed_data, dict):
                                continue
                            converted_seed = transform_keys(seed_data, prefix)
                            if converted_seed:
                                for ck in COMMON_KEYS:
                                    if ck in seed_data:
                                        converted_seed[ck] = seed_data[ck]
                                new_seeds.append(converted_seed)
                        new_data["all_seeds"] = new_seeds

                    if "train_config" in new_data and isinstance(new_data["train_config"], dict):
                        old_exp_name = new_data["train_config"].get("exp_name", args.target)
                        new_data["train_config"]["exp_name"] = f"{old_exp_name}-{mode}"

                        if "base_path" in new_data["train_config"] and isinstance(new_data["train_config"]["base_path"], str):
                            new_data["train_config"]["base_path"] = new_data["train_config"]["base_path"].replace(
                                args.target, f"{args.target}-{mode}"
                            )

                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    with open(dest_path, "w") as f:
                        json.dump(new_data, f, indent=4)

                except Exception as e:
                    print(f"Error while processing mode={mode} for {src_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
