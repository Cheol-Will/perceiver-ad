import os, json, glob
import pandas as pd

BASE_DIR = "results"
TRAIN_RATIOS = [0.1, 0.5, 1.0]
METRICS_CANON = ["AUC-ROC", "AUC-PR"]
METRIC_ALIAS = {
    "AUC-ROC": "AUC-ROC", "AUROC": "AUC-ROC", "AUCROC": "AUC-ROC",
    "AUC_ROC": "AUC-ROC", "auc_roc": "AUC-ROC",
    "AUC-PR": "AUC-PR", "AUCPR": "AUC-PR", "AUC_PR": "AUC-PR",
    "auc_pr": "AUC-PR",
}

def canon_metric_name(name: str):
    return METRIC_ALIAS.get(name, name)

def collect_results():
    rows = []
    # results/*/*/*/summary.json
    for path in glob.glob(os.path.join(BASE_DIR, "*", "*", "*", "summary.json")):
        parts = os.path.normpath(path).split(os.sep)
        # ... / results / <model_dir> / <dataset> / <train_ratio> / summary.json
        try:
            model_dir   = parts[-4]  # 
            dataset     = parts[-3]
            train_ratio = float(parts[-2])
        except Exception as e:
            print(f"[skip] path parse failed: {path} ({e})")
            continue

        try:
            with open(path, "r") as f:
                js = json.load(f)
        except Exception as e:
            print(f"[skip] cannot load json: {path} ({e})")
            continue

        mm = js.get("mean_metrics", {}) or {}
        metric_vals = {}
        for k, v in mm.items():
            ck = canon_metric_name(k)
            if ck in METRICS_CANON:
                metric_vals[ck] = v

        if not metric_vals:
            continue

        rows.append({
            "model": model_dir,
            "dataset": dataset,
            "train_ratio": train_ratio,
            **{m: metric_vals.get(m, None) for m in METRICS_CANON},
        })
    return rows

def convert_results_to_csv(results, save_csv=False, outdir="summary"):
    df_all = pd.DataFrame(results)

    # 
    for col in ["model", "dataset", "train_ratio"] + METRICS_CANON:
        if col not in df_all.columns:
            df_all[col] = pd.Series(dtype="float64" if col in METRICS_CANON else "object")

    # 6 tables (train_ratio × metric)
    dfs = {}
    for tr in TRAIN_RATIOS:
        for metric in METRICS_CANON:
            df_sub = (
                df_all[df_all["train_ratio"] == tr]
                .loc[:, ["model", "dataset", metric]]
                .sort_values(["model", "dataset"])
                .reset_index(drop=True)
            )

            key = f"ratio_{tr}_{metric.replace('-', '')}"
            dfs[key] = df_sub
            if save_csv:
                os.makedirs(outdir, exist_ok=True)
                df_sub.to_csv(os.path.join(outdir, f"{key}.csv"), index=False)
    return df_all, dfs

def make_pivots(dfs, save_csv=False, outdir="summary"):
    pivots = {}
    for tr in TRAIN_RATIOS:
        for metric in METRICS_CANON:
            key_src = f"ratio_{tr}_{metric.replace('-', '')}"
            if key_src not in dfs:
                continue
            df = dfs[key_src].copy()
            if df.empty:
                continue
            pivoted = (
                df.pivot(index="model", columns="dataset", values=metric)
                  .sort_index(axis=0)
                  .sort_index(axis=1)
            )
            pivoted.columns = pivoted.columns.str.lower()
            pivoted = pivoted[sorted(pivoted.columns)]
            # pivoted['AVG'] = pivoted.mean(axis=1)
            key_piv = f"ratio_{tr}_{metric.replace('-', '')}"
            pivots[key_piv] = pivoted
            if save_csv:
                os.makedirs(outdir, exist_ok=True)
                pivoted.to_csv(os.path.join(outdir, f"{key_piv}.csv"))
    return pivots

def main():
    data=[
        'arrhythmia',
        'breastw', 
        'cardio',
        'campaign',
        'cardiotocography', 
        # 'census',
        'glass',
        'ionosphere',
        'mammography', 
        # 'nslkdd',
        'hepatitis',
        'optdigits',
        'pendigits',
        'pima',
        'satellite',
        # 'shuttle',
        # 'satimage-2',
        'thyroid',
        'wbc',
        'wine',
    ]


    results = collect_results()
    df_all, dfs = convert_results_to_csv(results, save_csv=False)
    dfs = make_pivots(dfs, save_csv=False)
    keys = [
        # 'ratio_0.1_AUCROC', 
        # 'ratio_0.5_AUCROC', 
       # 'ratio_1.0_AUCROC', 
        # 'ratio_0.1_AUCPR', 
        # 'ratio_0.5_AUCPR', 
        'ratio_1.0_AUCPR'
    ]

    for k in keys:
        print(k)
        df = dfs[k][data].copy()
        # df = dfs[k].copy()
        df.loc[:, 'AVG'] = df.mean(axis=1)
        df = df.round(4)
        df = df.T

        df.columns = [c.replace("Perceiver", "Per") if "Perceiver" in c else c 
                  for c in df.columns]
        
        df.to_csv(f'metrics/{k}.csv')
        df.T.to_csv(f'metrics/{k}_T.csv')

        # cols = [c for c in df.columns if ("Per" not in c and 'RIN' not in c)]
        # df = df[cols].copy()

        print(df.T)
        print()

if __name__ == "__main__":
    main()