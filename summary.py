import os, json, glob
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
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
    dataset_properties = pd.read_csv("Data/dataset_mcm.csv")
    dataset_properties['Dataset'] = dataset_properties['Dataset'].str.lower()
    dataset_properties['Baseline (ratio)'] = dataset_properties['Anomaly'] / (dataset_properties['Anomaly'] + (dataset_properties['Samples'] // 2)) 
    dataset_properties = dataset_properties.set_index('Dataset')
    baseline = dataset_properties.loc[data, 'Baseline (ratio)']
    # print(dataset_properties)
    df.loc['Baseline (ratio)', data] = baseline
    df.iloc[-1, -1] = 0 # for avg rank column
    df.iloc[-1, -2] = np.mean(baseline) # for avg_auc column

    return df

def plot_avg_rank(df_with_rank, base, filename):
    rank_data = df_with_rank[['AVG_RANK']].sort_values(by='AVG_RANK', ascending=False)
    new_index = [
        'MemPAE' if 'MemPAE' in model_name  
        else 'PAE' if 'PAE' in model_name  
        else 'PDRL' if 'PDRL' in model_name 
        else model_name                    
        for model_name in rank_data.index
    ]
    rank_data.index = new_index

    plt.figure(figsize=(12, 8))

    ax = sns.barplot(x=rank_data.index, y=rank_data['AVG_RANK'], hue=rank_data.index, palette='viridis', legend=False)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    fontsize=12,
                    textcoords='offset points')

    plt.title(f'Model Average Rank ({filename})', fontsize=20)
    plt.xlabel('Model', fontsize=15)
    plt.ylabel('Average Rank', fontsize=15)
    plt.xticks(rotation=45, ha='right', fontsize=12) 
    plt.ylim(0, rank_data['AVG_RANK'].max() * 1.15) 
    plt.tight_layout() 

    save_path = f'metrics/{filename}_rank_plot.png'
    plt.savefig(save_path)
    print(f"Rank plot saved to {save_path}")
    plt.close() 

def render(
    pivots, # 
    data, # data to include
    models, # baseline models
    my_models, # ours
    base, # metric name
    add_avg_rank = False,
    is_sort = False,
    use_rank = False,  
    use_std = False, 
    use_baseline_pr = True,
    use_alias = False,
    is_temp_tune = False,
    is_synthetic = False,
    synthetic_type = None,
    is_plot = False,
):
    tr, metr = base.split('_')[1], base.split('_')[2]  # e.g., '1.0', 'AUCPR'
    k_mean = f"ratio_{tr}_{metr}_mean"
    k_std  = f"ratio_{tr}_{metr}_std"
    df_mean = pivots[k_mean][data].copy()
    df_std  = pivots[k_std][data].copy()

    # ordering
    first = [m for m in models if m in df_mean.index]
    rest  = [m for m in df_mean.index if m not in first]
    rest  = my_models
    order = first + rest
    df_mean = df_mean.loc[order]
    df_std  = df_std.loc[order]

    if is_temp_tune:
        row1_name = 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1'
        row2_name = 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.05'
        if is_synthetic:
            row1_name = 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.05'
            row2_name = 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1'
        new_row_name = 'MemPAE-ws-pos_query+token-d64-lr0.001-t-tune'

        mean_comparison = df_mean.loc[[row1_name, row2_name]]
        winner_indices = mean_comparison.idxmax()
        new_mean_row = mean_comparison.max()

        is_row1_winner = (winner_indices == row1_name)
        new_std_row = pd.Series(
            np.where(is_row1_winner, df_std.loc[row1_name], df_std.loc[row2_name]),
            index=df_std.columns
        )

        df_mean.loc[new_row_name] = new_mean_row
        df_std.loc[new_row_name] = new_std_row

        df_mean.drop([row1_name, row2_name], inplace=True)
        df_std.drop([row1_name, row2_name], inplace=True)

    df_mean.loc[:, 'AVG_AUC'] = df_mean.mean(axis=1, numeric_only=True)
    df_std.loc[:, 'AVG_AUC']  = df_std.mean(axis=1, numeric_only=True)

    if add_avg_rank:
        if is_synthetic:
            plot_name = f'synthetic_{synthetic_type}_{base}'
        else:
            plot_name = base
        df_mean = add_rank_columns(df_mean, is_sort=is_sort)
        if is_plot: 
            plot_avg_rank(df_mean, base, plot_name) 

    if use_rank:
        df_mean.loc[:, data] = df_mean.loc[:, data].rank(axis=0, ascending=False, method='average')

    
    df_render = df_mean.copy()
    if use_rank:
        target_cols = [col for col in df_render.columns if col not in data]
        # df_render.loc[:, target_cols] = df_render.loc[:, target_cols].round(4)
        df_render.loc[:, data] = df_render.loc[:, data].round(0)

    if use_std:
        df_render = _render_mean_pm_std(df_mean.round(4), df_std.round(4))

    if base == 'ratio_1.0_AUCPR':
        if use_baseline_pr and not use_rank and not use_std:
            df_render = add_baseline_pr(df_render, data)
            df_render.loc['Baseline (ratio)', data] = df_render.loc['Baseline (ratio)', data]
    if not use_std:
        df_render = df_render.round(4)

    if use_alias:
        aliases = {
            'global': 'G',
            'cluster': 'C',
            'local': 'L',
            'dependency': 'D',
            '_anomalies': '',
            'irrelevant_features': 'if',
            ###
            '32_shuttle_42': 'stl',
            '29_pima_42': 'pm',
            '38_thyroid_42': 'thy',
            '6_cardio_42': 'car',
            '31_satimage-2_42': 'sati',
            '18_ionosphere_42': 'ion',
            '4_breastw_42': 'bre',
            '45_wine_42': 'wi',
            '23_mammography_42': 'mam',
            '30_satellite_42': 'satl',
            '7_cardiotocography_42': 'cart',
            '13_fraud_42': 'fr',

            '5_campaign_42': 'camp',
            '9_census_42': 'cen',
            '14_glass_42': 'gls',
            '26_optdigits_42': 'opt',
            '42_WBC_42' : 'wbd',



        }
        def apply_aliases(column_name, aliases_dict):
            new_name = column_name
            for keyword, alias in aliases_dict.items():
                new_name = new_name.replace(keyword, alias)
            return new_name
        
        df_render = df_render.rename(
            columns=lambda c: apply_aliases(c, aliases)
        )
        df_render = df_render.round(4)


    df_render.index = [c.replace("MemPAE-ws-pos_query+token", "MP") if "MemPAE-ws-pos_query+token" in c else c
                        for c in df_render.index]

    os.makedirs('metrics', exist_ok=True)
    df_render.to_csv(f'metrics/{base}.csv')
    df_render.T.to_csv(f'metrics/{base}_T.csv')

    print(base)
    print(df_render)
    # print(df_render.T)
    print()

    return df_render


def main(args):
    keys = [
        # 'ratio_0.1_AUCROC', 'ratio_0.5_AUCROC', 
        # 'ratio_0.1_AUCPR', 'ratio_0.5_AUCPR',
        'ratio_1.0_AUCROC',
        'ratio_1.0_AUCPR',
    ]
    models=  [
        'IForest', 'LOF', 'OCSVM', 'ECOD', 'KNN', 'PCA',  # KNN: 0.6918, LOF: 0.6612
        'AutoEncoder', 
        'DeepSVDD', 'GOAD', 
        'NeuTraL', 'ICL', 'MCM', 'DRL',
        'Disent',
    ]

    data = [
        'arrhythmia', 'breastw', 'cardio', 'cardiotocography', 'glass',
        'ionosphere', 'pima', 'wbc', 'wine', 'thyroid',
        'optdigits', 'pendigits', 'satellite', 
        'campaign', 
        'mammography', 
        'satimage-2', # middle
        'nslkdd', # large 
        'fraud', # large
        'shuttle', # large
        'census', # large
    ]
    data.sort()

    my_models = [
        # 'Perceiver-d16-dcol0.5',
        # 'RIN-d16-dcol0.1',

        # 'MemAE-d64-lr0.05', # 0.6834    4.8750 (KNN: 4.1875)
        # 'MemAE-d256-lr0.001', # 0.6751
        # 'MemAE-d256-lr0.01-t0.1', # 0.6893    5.1875 (KNN: 4.0000)
        # 'MemAE-l2-d128-lr0.005', # 0.6744    4.8750 (KNN: 4.1250)
        # 'PDRL-ws-pos_query+token-d64-lr0.001',

        # 'PAE-ws-d64-lr0.001', # 0.6867    3.6875 # (SOTA! KNN: 4.3125)
        # 'PAE-ws-L6-d64-lr0.001', # SOTA!!!
        # 'PAE-ws-L2-d64-lr0.001', # 
        # 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1',
        
        'MemPAE-ws-pos_query-use_ent_score-ent0.001-d64-lr0.001',
        'MemPAE-ws-pos_query-use_ent_score-ent0.0002-d64-lr0.001',

        # 'MemPAE-ws-pos_query+token-L2-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-L3-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-L5-d64-lr0.001-t0.1',

        # 'MemPAE-ws-pos_query+token-np-L5-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-np-top1-L4-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-np-top1-L5-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-np-top5-L3-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-np-top5-L4-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-np-top5-L5-d64-lr0.001-t0.1', # 
        # 'MemPAE-ws-pos_query+token-use_ent_score-ent0.01-L5-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-use_ent_score-ent0.05-L5-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-use_ent_score-ent0.005-L5-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-use_ent_score-ent0.001-L5-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-use_ent_score-ent0.0005-L5-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-use_ent_score-ent0.0001-L5-d64-lr0.001-t0.1',


        #########################################
        'MemPAE-ws-pos_query+token-np-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-np-use_ent_score-ent0.001-L4-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-np-use_ent_score-ent0.001-L5-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-np-use_ent_score-ent0.001-L6-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-np-use_ent_score-ent0.001-L7-d64-lr0.001-t0.1',

        'MemPAE-ws-pos_query+token-np-use_ent_score-ent0.0005-L4-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-np-use_ent_score-ent0.0005-L5-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-np-use_ent_score-ent0.0005-L6-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-np-use_ent_score-ent0.0005-L7-d64-lr0.001-t0.1',

        'MemPAE-ws-pos_query+token-np-use_ent_score-ent0.0001-L4-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-np-use_ent_score-ent0.0001-L5-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-np-use_ent_score-ent0.0001-L6-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-np-use_ent_score-ent0.0001-L7-d64-lr0.001-t0.1',
        #########################################




        #########################################
        # 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.05',

        # 'MemPAE-ws-pos_query-ent0.001-d64-lr0.001',
        # 'MemPAE-ws-pos_query+token-ent0.001-L2-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-ent0.001-L3-d64-lr0.001-t0.1',
        # 'MemPAE-ws-attn-pos_query+token-L4-d64-lr0.001',

        # 'MemPAE-ws-pos_query+token-ent0.001-L5-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-ent0.0001-L5-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-L5-d64-lr0.005-t0.1',

        # 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.05',
        
        # 'MemPAE-ws-pos_query+token-np-L2-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-np-L3-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-np-L5-d64-lr0.001-t0.1',

        # 'MemPAE-ws-pos_query+token-np-top5-L3-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-np-top5-L4-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-np-top5-L5-d64-lr0.001-t0.1', # 

        # temperature: [0.05, 0.1]
        # 'MemPAE-ws-pos_query-d64-lr0.001-t0.1', #  0.6892    3.7500 (SOTA! KNN: 4.2500) # working on large data
        # 'MemPAE-ws-pos_query-d64-lr0.001-t0.05', #  0.6892    3.7500 (SOTA! KNN: 4.2500) # working on large data

        # 'MemPAE-ws-pos_query+token-d64-lr0.001',
        
        
        # 'MemPAE-ws-pos_query-ent0.0002-d64-lr0.001',
        # 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.01',
       
        # 'MemPAE-ws-pos_query-d64-lr0.001-t0.05', #  0.6892    3.7500 (SOTA! KNN: 4.2500) # working on large data
        # 'MemPAE-ws-pos_query-ent0.0002-d64-lr0.001',
        # 'MemPAE-ws-pos_query-ent0.001-d64-lr0.001',
        # 'MemPAE-ws-pos_query-use_ent_score-ent0.0002-d64-lr0.001',
        # 'MemPAE-ws-pos_query-use_ent_score-ent0.001-d64-lr0.001',

        # # 'PAE-ws-L2-d64-lr0.001',
        # 'MemPAE-ws-pos_query-L6-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query-L2-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query-ent0.0002-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query-ent0.001-d64-lr0.001-t0.1',

        # 'PAE-ws-pos_query-d64-lr0.001',
        # 'PAE-ws-d64-lr0.001-log', # N = log2(F) 

        # 'MemPAE-ws-d64-lr0.001', #  0.6892    3.7500 (SOTA! KNN: 4.2500) # working on large data
        # 'MemPAE-ws-d64-lr0.001', #  

        # 'MemPAE-ws-pos_query-ent0.05-d64-lr0.001',
        # 'MemPAE-ws-pos_query-ent0.01-d64-lr0.001',

        # 'MemPAE-ws-pos_query-ent0.05-d64-lr0.001-t0.1',
        # 'MemPAE-ws-ent0.05-d64-lr0.001',
        
        # 'MemPAE-ws-pos_query-ent0.01-d64-lr0.001-t0.1',
        # 'MemPAE-ws-ent0.1-d64-lr0.001',

        # 'PAE-ws-pos_query-d64-lr0.001',

        # 'PairMemPAE-ws-d32-lr0.001',
        # 'PairMemPAE-ws-d64-lr0.001',
        # 'PairMemPAE-ws-d32-lr0.005',
        # 'PairMemPAE-ws-d64-lr0.005',
        # 'TripletMemPAE-ws-d64-lr0.001',
        # 'TripletMemPAE-ws-d64-lr0.005',
        # 'TripletMemPAE-ws-d32-lr0.001',
        # 'TripletMemPAE-ws-d32-lr0.005', 
        # 'PAE-ws-d64-lr0.005', # 0.6867    3.6875 # (SOTA! KNN: 4.3125)
        # 'PAE-ws-d32-lr0.001', # 0.6867    3.6875 # (SOTA! KNN: 4.3125)
        # 'PAE-ws-d32-lr0.005', # 0.6867    3.6875 # (SOTA! KNN: 4.3125)
        # 'PAE-ws-d16-lr0.001', 
        # 'PAE-ws-d16-lr0.005', 
        # 'PAE-d64-lr0.001', # 0.6867    3.8125  (SOTA! KNN: 4.2500)
        # 'PAE-ws-pos_query-d64-lr0.001', # 0.6836    4.0000 (SOTA! KNN: 4.3125)
        # 'PAE-pos_query-d64-lr0.001', # Not done

        # 'MemPAE-ws-d64-lr0.001-smem',
        # 'MemPAE', # 0.6878    3.8125 (SOTA! KNN: 4.1875)
        # 'MemPAE-ws-d64-lr0.001', # 0.6878    3.7500 (SOTA! KNN: 4.2500)
        # 'MemPAE-ws-d64-lr0.0005', # 0.6878    3.7500 (SOTA! KNN: 4.2500)
        # 'MemPAE-ws-d64-lr0.002', # 0.6878    3.7500 (SOTA! KNN: 4.2500)
        # 'MemPAE-ws-d64-lr0.003', # 0.6878    3.7500 (SOTA! KNN: 4.2500)
        # 'MemPAE-ws-attn-d64-lr0.001', 
        # 'MemPAE-ws-l2-d64-lr0.001', 
        # 'MemPAE-ws-d64-lr0.005', # 0.6878    3.7500 (SOTA! KNN: 4.2500)
        # 'MemPAE-ws-d32-lr0.001', # 0.6878    3.7500 (SOTA! KNN: 4.2500)
        # 'MemPAE-ws-d32-lr0.005', # 0.6878    3.7500 (SOTA! KNN: 4.2500)
        # 'MemPAE-ws-d64-lr0.001-t0.1', #  0.6892    3.7500 (SOTA! KNN: 4.2500)
        # 'MemPAE-ws-pos_query-d64-lr0.001', # working on
        # 'MemPAE-ws-pos_query-d64-lr0.001-thres0.0025',

        # 'PAEKNN-ws-d64-lr0.001',
        # 'PVAE-ws-d64-lr0.001',
        # 'PVAE-ws-d64-lr0.001-beta0.25',
        # 'PVAE-ws-d32-lr0.001',
        # 'PVAE-ws-d32-lr0.001-besta0.25',

        # 'PVQVAE-ws-d64-lr0.001-beta1-vq',
        # 'PVQVAE-ws-d64-lr0.001-beta1.0',
        # 'PVQVAE-ws-d32-lr0.001-beta1.0',
        # 'PVQVAE-ws-d64-lr0.001-beta0.25',
    ]

    results = collect_results()
    df_all, dfs = convert_results_to_csv(results, save_csv=False)
    pivots = make_pivots(dfs, save_csv=False)

    for base in keys:
        render(pivots, data, models, my_models, base, 
               add_avg_rank=True, use_rank=False, use_std=False, 
               use_baseline_pr=True, is_temp_tune=False, is_sort=False)

    models = [
        'IForest', 'LOF', 'OCSVM', 'ECOD', 'KNN', 'PCA',  # KNN: 0.6918, LOF: 0.6612
        'AutoEncoder', 
        'DeepSVDD', 'GOAD', 
        'NeuTraL', 'ICL', 'MCM', 'DRL',
        'Disent',
    ]
    my_models = [

        # 'PDRL-ws-pos_query+token-d64-lr0.001',

        # 'PAE-ws-d64-lr0.001', # 0.6867    3.6875 # (SOTA! KNN: 4.3125)
        'PAE-ws-L6-d64-lr0.001', # 0.6867    3.6875 # (SOTA! KNN: 4.3125)
        # 'MemPAE-ws-d64-lr0.001', # 0.6878    3.7500 (SOTA! KNN: 4.2500)
        # 'MemPAE-ws-pos_query-d64-lr0.001-t0.1', # 0.6878    3.7500 (SOTA! KNN: 4.2500)
        # 'MemPAE-ws-pos_query-d64-lr0.001', # 0.6878    3.7500 (SOTA! KNN: 4.2500)
        'MemPAE-ws-pos_query+token-d64-lr0.001-t0.05',
        # 'MemPAE-ws-pos_query-d64-lr0.001-t0.05',
        'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-np-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-np-L6-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-L6-d64-lr0.001-t0.05',
        
    ]


    # todo: make name shorter.
    if args.synthetic:
        
        models = [
            'IForest', 
            'LOF', 
            'OCSVM', 
            'ECOD', 
            'KNN', 
            'PCA',  # KNN: 0.6918, LOF: 0.6612
            'AutoEncoder', 
            # 'DeepSVDD', 'GOAD', 
            # 'NeuTraL', 'ICL', 
            'MCM', 'DRL',
            'Disent',
        ]

        # success case: cardio, sat (maybe)
        dataname_list = [

            # only care about dependency anomaly
            # note that local anomaly requires 
            # some density estimation or retrieval modules.
            # '45_wine', # 0.1k
            # '14_glass',
            # '42_wbc',


            # above nope small data. not good at all
            '18_ionosphere', # 0.3k   
            '4_breastw', # 0.6k
            '29_pima', # 0.7k
            '6_cardio', # 1.8k
            '7_cardiotocography', # 2k
            '38_thyroid', # 3k
            '26_optdigits',
            '31_satimage-2', # 5k
            '30_satellite', # 6k
            '23_mammography', # 11k
            '5_campaign',
            '32_shuttle', # 49k

            # here cut
            # below nope
            #################  
            # '13_fraud', # good for local anomaly setting.
            # '9_census',
            #################      


            #################      
            ]

        
        # prefix_list = [
        #     # 'cluster_anomalies_',
        #     # 'global_anomalies_',
        #     'local_anomalies_',
        #     # 'dependency_anomalies_/',
        #     # f'{args.synthetic_type}_anomalies_',
        # ]
        anomaly_type = 'local_anomalies_'
        # anomaly_type = 'dependency_anomalies_'

        irrelevant_features_list = [
            '',  
            # 'irrelevant_features_0.1_',
            # 'irrelevant_features_0.3_',
            # 'irrelevant_features_0.5_',
        ]

        suffix = '_42'
        synthetic_data = []

        for dataname in dataname_list:
            for feature in irrelevant_features_list:
                file_name = f"{anomaly_type}{feature}{dataname}{suffix}"
                synthetic_data.append(file_name)
        # synthetic_data = [
        #     'shuttle',
        #     # '32_shuttle_irrelevant_features_0.1_42',
        #     # '32_shuttle_irrelevant_features_0.3_42',
        #     '32_shuttle_irrelevant_features_0.5_42',

        #     'breastw',
        #     # '4_breastw_irrelevant_features_0.1_42',
        #     # '4_breastw_irrelevant_features_0.3_42',
        #     '4_breastw_irrelevant_features_0.5_42',

        #     'cardio',
        #     # '6_cardio_irrelevant_features_0.1_42',
        #     # '6_cardio_irrelevant_features_0.3_42',
        #     '6_cardio_irrelevant_features_0.5_42',

        #     'wbc',
        #     # '42_wbc_irrelevant_features_0.1_42',
        #     # '42_wbc_irrelevant_features_0.3_42',
        #     '42_wbc_irrelevant_features_0.5_42',

        #     'wine',
        #     # '45_wine_irrelevant_features_0.1_42',
        #     # '45_wine_irrelevant_features_0.3_42',
        #     '45_wine_irrelevant_features_0.5_42',

        # ]

        for base in keys:
            df_render = render(pivots, synthetic_data, models, my_models, base,
                add_avg_rank=True, use_rank=False, use_std=True, use_baseline_pr=False, 
                use_alias=True, is_temp_tune=True, is_synthetic=True, synthetic_type=anomaly_type)
            # df_render['loc_diff_st'] = df_render['loc_st'] - df_render['loc_if_0.5_st'] 
            # df_render['dep_diff_st'] = df_render['dep_st'] - df_render['dep_if_0.5_st']
            # df_render['dep_diff_pm'] = df_render['dep_pm'] - df_render['dep_if_0.5_pm'] 
            # df_render['loc_diff_pm'] = df_render['loc_pm'] - df_render['loc_if_0.5_pm'] 

            # print(df_render)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--synthetic_type', type=str, default='dependency')
    args = parser.parse_args()
    main(args)