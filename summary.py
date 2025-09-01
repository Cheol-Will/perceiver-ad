import os, json, glob
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)

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
            model_dir   = parts[-4]
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

        # collect mean/std from all_seeds if available; else fallback to mean_metrics
        metric_means, metric_stds = {}, {}
        seeds = js.get("all_seeds", None)
        if seeds and isinstance(seeds, list) and len(seeds) > 0:
            # gather per-metric list
            acc = {m: [] for m in METRICS_CANON}
            for rec in seeds:
                for k, v in rec.items():
                    ck = canon_metric_name(k)
                    if ck in METRICS_CANON:
                        acc[ck].append(v)
            for m in METRICS_CANON:
                vals = np.asarray(acc[m], dtype=float) if len(acc[m]) > 0 else None
                if vals is not None:
                    metric_means[m] = float(np.mean(vals))
                    metric_stds[m]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        else:
            # fallback to provided mean_metrics
            mm = (js.get("mean_metrics", {}) or {})
            for k, v in mm.items():
                ck = canon_metric_name(k)
                if ck in METRICS_CANON:
                    metric_means[ck] = float(v)
                    metric_stds[ck]  = np.nan  # std unknown

        if not metric_means:
            continue

        row = {"model": model_dir, "dataset": dataset, "train_ratio": train_ratio}
        for m in METRICS_CANON:
            row[f"{m}_mean"] = metric_means.get(m, np.nan)
            row[f"{m}_std"]  = metric_stds.get(m,  np.nan)
        rows.append(row)
    return rows

def convert_results_to_csv(results, save_csv=False, outdir="summary"):
    df_all = pd.DataFrame(results)

    # ensure columns exist
    need_cols = ["model", "dataset", "train_ratio"]
    for m in METRICS_CANON:
        need_cols += [f"{m}_mean", f"{m}_std"]
    for col in need_cols:
        if col not in df_all.columns:
            df_all[col] = pd.Series(dtype="float64" if col.endswith(("_mean","_std")) else "object")

    # build per (train_ratio, metric, stat)
    dfs = {}
    for tr in TRAIN_RATIOS:
        for metric in METRICS_CANON:
            for stat in ["mean", "std"]:
                colname = f"{metric}_{stat}"
                df_sub = (
                    df_all[df_all["train_ratio"] == tr]
                    .loc[:, ["model", "dataset", colname]]
                    .sort_values(["model", "dataset"])
                    .reset_index(drop=True)
                )
                key = f"ratio_{tr}_{metric.replace('-', '')}_{stat}"
                dfs[key] = df_sub
                if save_csv:
                    os.makedirs(outdir, exist_ok=True)
                    df_sub.to_csv(os.path.join(outdir, f"{key}.csv"), index=False)
    return df_all, dfs

def make_pivots(dfs, save_csv=False, outdir="summary"):
    pivots = {}
    for tr in TRAIN_RATIOS:
        for metric in METRICS_CANON:
            for stat in ["mean", "std"]:
                key_src = f"ratio_{tr}_{metric.replace('-', '')}_{stat}"
                if key_src not in dfs:
                    continue
                df = dfs[key_src].copy()
                if df.empty:
                    continue
                val_col = f"{metric}_{stat}"
                pivoted = (
                    df.pivot(index="model", columns="dataset", values=val_col)
                      .sort_index(axis=0)
                      .sort_index(axis=1)
                )
                pivoted.columns = pivoted.columns.str.lower()
                pivoted = pivoted[sorted(pivoted.columns)]
                key_piv = f"ratio_{tr}_{metric.replace('-', '')}_{stat}"
                pivots[key_piv] = pivoted
                if save_csv:
                    os.makedirs(outdir, exist_ok=True)
                    pivoted.to_csv(os.path.join(outdir, f"{key_piv}.csv"))
    return pivots

def add_rank_columns(
    df_mean: pd.DataFrame,
    tie_method: str = "average",
    is_sort: bool = False,
):
    
    # higher_is_better
    ranks = df_mean.rank(axis=0, ascending=False, method=tie_method)
    avg_rank = ranks.mean(axis=1, skipna=True).rename('AVG_RANK')

    df_with_avg_rank = df_mean.copy()
    df_with_avg_rank['AVG_RANK'] = avg_rank

    if is_sort:
        df_with_avg_rank = df_with_avg_rank.sort_values(by=['AVG_RANK'])

    return df_with_avg_rank

def add_tire_colums(
    df_mean: pd.DataFrame,
    df_std: pd.DataFrame,
) -> pd.DataFrame:
    dfm = df_mean.copy()
    cols = [c for c in dfm.columns if c in df_std.columns] # extract only dataset column (excludes AVG_RANK columns)

    def _tier_one_column(s_mean: pd.Series, s_std: pd.Series) -> pd.Series:
        ascending = False
        s_std = s_std.fillna(0.0)

        order = s_mean.sort_values(ascending=ascending, na_position="last").index.tolist()

        tiers = {idx: np.nan for idx in s_mean.index}
        ref_idx = None
        for idx in order:
            if not pd.isna(s_mean.loc[idx]):
                ref_idx = idx
                break
        if ref_idx is None:
            return pd.Series(tiers)

        curr_tier = 1
        ref_mean = float(s_mean.loc[ref_idx])
        ref_std  = float(s_std.loc[ref_idx])
        tiers[ref_idx] = curr_tier

        for idx in order[order.index(ref_idx)+1:]:
            m = s_mean.loc[idx]
            if pd.isna(m):
                tiers[idx] = np.nan
                continue
            sd = float(s_std.loc[idx])
            same_tier = (float(m) <= ref_mean + ref_std)
            if same_tier:
                tiers[idx] = curr_tier
            else:
                curr_tier += 1
                ref_mean, ref_std = float(m), sd
                tiers[idx] = curr_tier

        return pd.Series(tiers)

    tier_cols = []
    for c in cols:
        tiers = _tier_one_column(dfm[c], df_std[c])
        tcol = f"{c}_tier"
        dfm[tcol] = tiers
        tier_cols.append(tcol)

    dfm['AVG_TIER'] = dfm[tier_cols].mean(axis=1, skipna=True)
    dfm.drop(columns=tier_cols, inplace=True) 


    return dfm


def _render_mean_pm_std(df_mean: pd.DataFrame, df_std: pd.DataFrame, digits=4) -> pd.DataFrame:
    # align indexes/columns
    df_std = df_std.reindex(index=df_mean.index, columns=df_mean.columns)
    def fmt(m, s):
        if pd.isna(m):
            return ""
        if pd.isna(s):
            return f"{m:.{digits}f}"
        return f"{m:.{digits}f} ± {s:.{digits}f}"
    return pd.DataFrame(
        np.vectorize(fmt)(df_mean.values, df_std.values),
        index=df_mean.index, columns=df_mean.columns
    )

def add_baseline_pr(df, data):
    dataset_properties = pd.read_csv("Data/dataset_properties.csv")
    dataset_properties['Data'] = dataset_properties['Data'].str.lower()
    dataset_properties['Baseline (ratio)'] = dataset_properties['Anomalies'] / (dataset_properties['Samples'] // 2)
    dataset_properties = dataset_properties.set_index('Data')
    baseline = dataset_properties.loc[data, 'Baseline (ratio)']

    df.loc['Baseline (ratio)', data] = baseline
    df.iloc[-1, -1] = 0
    df.iloc[-1, -2] = np.mean(baseline)

    return df


def main():
    keys = [
        # 'ratio_0.1_AUCROC', 'ratio_0.5_AUCROC', 
        # 'ratio_0.1_AUCPR', 'ratio_0.5_AUCPR',
        'ratio_1.0_AUCROC',
        'ratio_1.0_AUCPR',
    ]

    data = [
        'arrhythmia', 'breastw', 'cardio', 
        # 'census',
        'campaign', 'cardiotocography', 'glass', 'ionosphere', 'mammography', 
        # 'nslkdd',
        # 'hepatitis', 
        'optdigits', 'pima', 'pendigits', 'satellite', 
        # 'staimage-2', 'shuttle'
        'thyroid', 'wbc', 'wine',
    ]

    models=  [
        'IForest', 'LOF', 'OCSVM', 'ECOD', 'KNN', 'PCA',  # KNN: 0.6918, LOF: 0.6612
        'AutoEncoder', 'DeepSVDD', 'GOAD', 
        'NeuTraL', 'ICL', 'MCM', 'DRL',
        # 'Disent',
    ]
    my_models = [
        # 'Perceiver-d16-dcol0.5',
        # 'RIN-d16-dcol0.1',

        # 'MemAE-d64-lr0.05', # 0.6751
        # 'MemAE-d256-lr0.001', # 0.6751
        # 'MemAE-d256-lr0.01-t0.1', # 0.6818
        # 'MemAE-l2-d128-lr0.005', # 0.6693

        'PAE-ws-d64-lr0.001', # 0.6867    3.5625 # (SOTA! KNN: 4.1875)
        # 'PAE-d64-lr0.001', # 0.6867    3.6875  (SOTA! KNN: 4.1250)
        # 'PAE-ws-pos_query-d64-lr0.001', # 0.6836    3.8750 (SOTA! KNN: 4.3125)

        # 'PAE-pos_query-d64-lr0.001', # Not done


        # 'MemPAE-d64-lr0.001', # 0.6878    3.8125 (SOTA! KNN: 4.1875)
        # 'MemPAE-ws-d64-lr0.001', # 0.6878    3.8125 (SOTA! KNN: 4.1875)
        # 'MemPAE-ws-d64-lr0.001-t0.1', # 0.6842    4.2500 (close KNN: 4.0625)
        # 'MemPAE-ws-pos_query-d64-lr0.001-t0.1', #  0.6892    3.8750 (SOTA! KNN: 4.2500)
        
        # 'MemPAE-ws-pos_query-d64-lr0.001 # Not done
    ]


    results = collect_results()
    df_all, dfs = convert_results_to_csv(results, save_csv=False)
    pivots = make_pivots(dfs, save_csv=False)

    for base in keys:
        tr, metr = base.split('_')[1], base.split('_')[2]  # e.g., '1.0', 'AUCPR'
        k_mean = f"ratio_{tr}_{metr}_mean"
        k_std  = f"ratio_{tr}_{metr}_std"
        print(base)
        df_mean = pivots[k_mean][data].copy()
        df_std  = pivots[k_std][data].copy()
        df_mean.loc[:, 'AVG_AUC'] = df_mean.mean(axis=1, numeric_only=True)
        df_std.loc[:, 'AVG_AUC']  = df_std.mean(axis=1, numeric_only=True)

        # ordering
        first = [m for m in models if m in df_mean.index]
        rest  = [m for m in df_mean.index if m not in first]
        rest  = my_models
        order = first + rest

        # filtering
        order = [col for col in order if 'MultiMemAE' not in col]
        order = [col for col in order if 'RINMLP' not in col]
        order = [col for col in order if 'PAE-d8' not in col]

        # order = [col for col in order if 'MemAE-l2' not in col]
        # order = [col for col in order if 'PAE' in col]

        df_mean = df_mean.loc[order]
        df_std  = df_std.loc[order]
        
        df_mean = add_rank_columns(df_mean, is_sort=False)
        # df_mean = add_tire_colums(df_mean, df_std)

        if base == 'ratio_1.0_AUCPR':
            df_mean = add_baseline_pr(df_mean, data)
            pass

        df_render = df_mean.round(4)
        df_render = _render_mean_pm_std(df_mean.round(4), df_std.round(4))
        # df_render.columns = [c.replace("Perceiver", "Per") if "Perceiver" in c else c
        #                      for c in df_render.columns]

        os.makedirs('metrics', exist_ok=True)
        df_render.to_csv(f'metrics/{base}.csv')
        df_render.T.to_csv(f'metrics/{base}_T.csv')

        # print(df_render.T)
        print(df_render)
        print()

if __name__ == "__main__":
    main()