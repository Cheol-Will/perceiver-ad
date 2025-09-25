import os, json, glob
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', None)

BASE_DIR = "results"
TRAIN_RATIOS = [0.1, 0.3, 0.5, 0.8, 1.0]
TRAIN_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]
METRICS_CANON = ["AUC-ROC", "AUC-PR", 'f1']
METRIC_ALIAS = {
    "AUC-ROC": "AUC-ROC", "AUROC": "AUC-ROC", "AUCROC": "AUC-ROC",
    "AUC_ROC": "AUC-ROC", "auc_roc": "AUC-ROC",
    "AUC-PR": "AUC-PR", "AUCPR": "AUC-PR", "AUC_PR": "AUC-PR",
    "auc_pr": "AUC-PR",
    'f1': 'f1'
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
    is_print = True,
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

    # df_mean.loc['MemPAE-ws-pos_query+token-L6-d64-lr0.001-t0.1', ''] = 0.975 # current
    # df_mean.loc['MemPAE-ws-pos_query+token-latent_ratio8.0-d64-lr0.001-t0.1', 'census'] = 0.24 # current


        # 'MemPAE-ws-pos_query+token-d64-lr0.001', # t=1
        # 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1', # this is final
        # 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.01', # this is final
        # 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.05',



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


    df_render.index = [c.replace("pos_query+token", "pos") if "pos_query+token" in c else c
                        for c in df_render.index]
                        
    df_render.index = [c.replace("MemPAE-ws-cross_attn", "MPCA") if "MemPAE-ws-cross_attn" in c else c
                        for c in df_render.index]

    os.makedirs('metrics', exist_ok=True)
    if is_synthetic:
        file_name = f'{base}_{synthetic_type}'
    else:
        file_name = base
    df_render.to_csv(f'metrics/{file_name}.csv')
    df_render.T.to_csv(f'metrics/{file_name}_T.csv')
    print(f"file saved in {file_name}")

    if is_print:
        print(base)
        print(df_render)
        # print(df_render.T)
        print()

    

    return df_render


def render_hp(pivots):
    data = [
        'arrhythmia', 'breastw', 'cardio', 'cardiotocography', 'glass',
        'ionosphere', 'pima', 'wbc', 'wine', 'thyroid',
        'optdigits', 'pendigits', 'satellite', 'campaign', 'mammography', 
        'satimage-2', 'nslkdd', 'fraud', 'shuttle', 'census', # large
    ]
    data.sort()
    models = ['KNN']
    my_models = [
        'MemPAE-ws-pos_query+token-latent_ratio0.5-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-latent_ratio2.0-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-latent_ratio4.0-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-latent_ratio8.0-d64-lr0.001-t0.1',
 
        'MemPAE-ws-pos_query+token-memory_ratio0.5-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-large_mem-L4-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-memory_ratio2.0-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-memory_ratio4.0-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-memory_ratio8.0-d64-lr0.001-t0.1',
       
 
    ]
    base = 'ratio_1.0_AUCPR'
    render(pivots, data, models, my_models, base, 
            add_avg_rank=True, use_rank=False, use_std=False, 
            use_baseline_pr=True, is_temp_tune=False, is_sort=False, is_plot=True)

def render_train_ratio(pivots, print_summary = False):
    # '23_mammography', # 11k: 5 numeric and 1 binary -------------
    #     '30_satellite', # 6k: image 
    #     '31_satimage-2', # 5k: image 
    #     '26_optdigits', # 5k: image
    #     '38_thyroid', # 3k: medical: 1 catgorical and 5 numeric -------------
    #     '7_cardiotocography', # -------------
    #     '18_ionosphere', # 0.3k: frequency and purse data
    #     '6_cardio', # 1.8k -------------
    #     '29_pima', # 0.7k
    #     '4_breastw', # 0.6k
    
    
    data = [
        # 'arrhythmia', 
        # 'breastw',
        # 'cardio', 
        # 'cardiotocography', 
        # 'mammography', 
        # 'glass',
        # 'ionosphere',
        # 'wbc',
        # 'wine',
        # 'pima', 
        # 'pendigits', 
        # 'thyroid',
        # 'shuttle', # all good.
        # 'satimage-2', 
        # 'satellite',
        # 'optdigits',
        'campaign', # too slow
        'nslkdd',
        'fraud',
        'census',
    ]
    models = [
        # 'KNN',
        'MCM',
        'DRL',
        'Disent',
    ]
    my_models = [
        'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1',
    ]
    keys = [
        # 'ratio_0.1_AUCROC', 'ratio_0.5_AUCROC', 
        # 'ratio_0.1_AUCPR', 'ratio_0.5_AUCPR',
        # 'ratio_1.0_AUCROC',
        'ratio_0.2_AUCPR',
        'ratio_0.4_AUCPR',
        'ratio_0.6_AUCPR',
        'ratio_0.8_AUCPR',
        'ratio_1.0_AUCPR',
    ]    

    # First, collect all dataframes for each ratio
    dict_df = {}
    for base in keys:
        df = render(pivots, data, models, my_models, base, 
                add_avg_rank=False, use_rank=False, use_std=False, 
                use_baseline_pr=False, is_temp_tune=False, is_sort=False, is_plot=False)
        dict_df[base] = df # consists of row=model, column=data

    all_models = dict_df[keys[0]].index.tolist()
    dataset_results = {}
    
    for dataset in data:
        ratio_labels = [key.replace('ratio_', '').replace('_AUCPR', '') for key in keys]
        model_vs_ratio_df = pd.DataFrame(index=all_models, columns=ratio_labels)
        for i, base_key in enumerate(keys):
            ratio_label = ratio_labels[i]
            if dataset in dict_df[base_key].columns:
                model_vs_ratio_df[ratio_label] = dict_df[base_key][dataset]
        
        dataset_results[dataset] = model_vs_ratio_df
        
        print(f"\nDataset: {dataset.upper()}")
        print("="*60)
        print("Model vs Training Ratio (AUCPR)")
        print(model_vs_ratio_df.round(4))
        
        os.makedirs('metrics/train_ratio', exist_ok=True)
        model_vs_ratio_df.to_csv(f'metrics/train_ratio/{dataset}_model_vs_ratio.csv')
    
    if print_summary : 
        print("\n" + "="*80)
        print("BEST MODEL PER RATIO PER DATASET")
        print("="*80)
        
        best_models_summary = {}
        for dataset in data:
            best_models_summary[dataset] = {}
            df = dataset_results[dataset]
            
            print(f"\n{dataset.upper()}:")
            for ratio in df.columns:
                if not df[ratio].isna().all():  # Check if column has any values
                    best_model = df[ratio].idxmax()
                    best_score = df[ratio].max()
                    best_models_summary[dataset][ratio] = (best_model, best_score)
                    print(f"  Ratio {ratio}: {best_model} ({best_score:.4f})")
        
    return dataset_results

def render_train_ratio_average(pivots):
    data = [
        # 'arrhythmia', 
        # 'breastw',
        # 'cardio', 
        # 'cardiotocography', 
        # 'mammography', 
        # 'glass',
        # 'ionosphere',
        # 'wbc',
        # 'wine',
        # 'pima', 
        # 'pendigits', 
        # 'thyroid',
        # 'shuttle',
        # 'satimage-2', 
        # 'satellite',
        # 'optdigits',
        'campaign',
        'nslkdd',
        # 'fraud',
        'census',
    ]
    models = [
        'MCM',
        'DRL',
        'Disent',
    ]
    my_models = [
        'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1',
    ]
    keys = [
        'ratio_0.2_AUCPR',
        'ratio_0.4_AUCPR',
        'ratio_0.6_AUCPR',
        'ratio_0.8_AUCPR',
        'ratio_1.0_AUCPR',
    ]    

    # 각 ratio별로 데이터프레임 수집
    dict_df = {}
    for base in keys:
        df = render(pivots, data, models, my_models, base, 
                add_avg_rank=False, use_rank=False, use_std=False, 
                use_baseline_pr=False, is_temp_tune=False, is_sort=False, is_plot=False)
        dict_df[base] = df

    all_models = dict_df[keys[0]].index.tolist()
    ratio_labels = [key.replace('ratio_', '').replace('_AUCPR', '') for key in keys]
    
    # 전체 평균 계산을 위한 데이터프레임 생성
    average_df = pd.DataFrame(index=all_models, columns=ratio_labels)
    
    for i, base_key in enumerate(keys):
        ratio_label = ratio_labels[i]
        # 각 ratio에서 모든 데이터셋에 대한 평균 계산
        average_df[ratio_label] = dict_df[base_key].mean(axis=1)
    
    print("="*60)
    print("OVERALL AVERAGE - Model vs Training Ratio (AUCPR)")
    print("="*60)
    print(average_df.round(4))
    
    # 저장
    os.makedirs('metrics/train_ratio', exist_ok=True)
    average_df.to_csv('metrics/train_ratio/overall_average_model_vs_ratio.csv')
    
    print(f"\nOverall average results saved to: metrics/train_ratio/overall_average_model_vs_ratio.csv")
    
    return average_df


def render_ours_on_npt(pivots):
    data = [
        'wine',
        # 'lympho',
        'glass',
        # 'vertebral',
        'wbc',
        # 'ecoli',
        'ionosphere',
        'arrhythmia',
        'breastw',
        'pima',
        # 'vowels',
        # 'letter',
        'cardio',
        # 'seismic',
        # 'musk',
        # 'speech',
        'thyroid',
        # 'abalone',
        'optdigits',
        'satimage-2',
        'satellite',
        'pendigits',
        'annthyroid',
        # 'mnist',
        'mammography',
        'shuttle',
        # 'forest_cover',
        'campaign',
        'fraud',
        # 'backdoor', # we got 5 remainig datasets.
    ]
    models = []
    my_models = [
        'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query-d32-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-d32-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-large_mem-L4-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-memory_ratio4.0-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-memory_ratio2.0-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-memory_ratio0.5-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-latent_ratio0.5-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-latent_ratio2.0-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-latent_ratio4.0-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query-L2-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-L3-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-L5-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query-L6-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-L4-d64-lr0.001-t0.2',
        'MemPAE-ws-pos_query+token-L5-d64-lr0.005-t0.1',
        'MemPAE-ws-pos_query+token-np-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-np-L2-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-np-L3-d64-lr0.001-t0.1',

    ]
    npt_aucroc = {
        "wine": 96.6,
        # "lympho": 99.9,
        "glass": 82.8,
        # "vertebral": 54.6,
        "wbc": 96.3,
        # "ecoli": 88.7,
        "ionosphere": 97.4,
        "arrhythmia": 80.1,
        "breastw": 98.6,
        "pima": 73.4,
        # "vowels": 99.3,
        # "letter": 96.1,
        "cardio": 94.7,
        # "seismic": 69.8,
        # "musk": 100,
        # "speech": 54.3,
        "thyroid": 97.8,
        # "abalone": 91.6,
        "optdigits": 97.5,
        "satimage-2": 99.9,
        "satellite": 80.3,
        "pendigits": 99.9,
        "annthyroid": 86.7,
        # "mnist": 94.4,
        "mammography": 88.6,
        # "mullcross": 100,
        "shuttle": 99.8,
        # "forest_cover": 95.8,
        "campaign": 79.1,
        "fraud": 95.7,
        # "backdoor": 95.2
    }
    npt_f1 = {
        "wine": 72.5,
        # "lympho": 94.2,
        "glass": 26.2,
        # "vertebral": 20.3,
        "wbc": 67.3,
        # "ecoli": 77.7,
        "ionosphere": 92.7,
        "arrhythmia": 60.4,
        "breastw": 95.7,
        "pima": 68.8,
        # "vowels": 88.7,
        # "letter": 71.4,
        "cardio": 78.1,
        # "seismic": 26.2,
        # "musk": 100,
        # "speech": 9.3,
        "thyroid": 77.0,
        # "abalone": 59.7,
        "optdigits": 62,
        "satimage-2": 94.8,
        "satellite": 74.6,
        "pendigits": 92.5,
        "annthyroid": 57.7,
        # "mnist": 43.6,
        "mammography": 43.6,
        # "mullcross": 100,
        "shuttle": 98.2,
        # "forest_cover": 58,
        "campaign": 49.8,
        "fraud": 58.1,
        # "backdoor": 84.1,
    }    
    npt_df = pd.DataFrame([npt_aucroc.values()], index=['NPT-AD'], columns=npt_aucroc.keys())
    npt_df = npt_df/100
    base = 'ratio_1.0_AUCROC'
    # base = 'ratio_1.0_f1'
    pae_df = render(pivots, data, models, my_models, base, 
            add_avg_rank=False, use_rank=False, use_std=False, 
            use_baseline_pr=False, is_temp_tune=False, is_sort=False, is_plot=False)
    pae_df = pae_df.max(axis=0).to_frame().T
    # print(pae_df)
    merged_df = merged_df = pd.concat([npt_df, pae_df])
    merged_df = merged_df.dropna(axis=1)
    merged_df['AVG_AUC'] = merged_df.mean(axis=1)
    print(merged_df)

    return 

def main(args):
    keys = [
        # 'ratio_0.1_AUCROC', 'ratio_0.5_AUCROC', 
        # 'ratio_0.1_AUCPR', 'ratio_0.5_AUCPR',
        'ratio_1.0_AUCROC',
        'ratio_1.0_AUCPR',
        # 'ratio_0.8_AUCPR',
        # 'ratio_0.5_AUCPR',
    ]
    models=  [
        'IForest', 'LOF', 'OCSVM', 'ECOD', 'KNN', 'PCA',  
        'DeepSVDD', 'GOAD', 'NeuTraL', 'ICL', 'MCM', 'DRL', 'Disent',
    ]
    data = [
        'arrhythmia', 'breastw', 'cardio', 'cardiotocography', 'glass',
        'ionosphere', 'pima', 'wbc', 'wine', 'thyroid', 'optdigits', 'pendigits', 'satellite', 
        'campaign', 'mammography', 'satimage-2', 'nslkdd', 'fraud', 'shuttle', 'census',
    ]

    data.sort()

    my_models = [
        ##################################################################################
        # 'PDRL-ws-pos_query+token-d64-lr0.001',
        # 'MemAE-d64-lr0.001-t0.1'
        # 'MCMPAE-ws-pos_query+token-d32-lr0.001',
        # 'MCMPAE-ws-pos_query+token-d64-lr0.001',
        # 'MCMPAE-ws-pos_query+token-d64-lr0.005',
        ##################################################################################
        # PAE
        'PAE-ws-pos_query+token-d64-lr0.001', # Final architecture for PAE
        ##################################################################################
        # L: depth
        # 'MemPAE-ws-pos_query+token-L0-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-L2-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1', # this is final
        # 'MemPAE-ws-pos_query+token-L6-d64-lr0.001-t0.1',
        ##################################################################################
        # temperature
        # 'MemPAE-ws-pos_query+token-d64-lr0.001', # t=1
        # 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.5', # 
        'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1', # Ours
        # 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.05', # done
        # 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.01', # (working on)

        ##################################################################################
        # 'MemPAE-ws-pos_query+token-mlp_dec-d64-lr0.001-t0.1',
        # 'MemPAE-ws-pos_query+token-mlp_enc-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-mlp_dec_mixer-d64-lr0.001-t0.1',
        'MemPAE-ws-pos_query+token-mlp_enc_mixer-d64-lr0.001-t0.1',
        ##################################################################################
        # "MemPAE-ws-pos_query+token-d64-lr0.001-t0.1",
        # "MemPAE-ws-global_query-d64-lr0.001-t0.1",
        # "MemPAE-ws-d64-lr0.001-t0.1",
        # "MemPAE-ws-global_token+d64-lr0.001-t0.1",
        # "MemPAE-ws-local+d64-lr0.001-t0.1",
        # "MemPAE-ws-pos_query-d64-lr0.001-t0.1",
        # queyr token
        # "MemPAE-ws-pos_query+token-d64-lr0.001-t0.1",
        # "MemPAE-ws-pos_query-d64-lr0.001-t0.1",
        # "MemPAE-ws-d64-lr0.001-t0.1",
        
        
        
        # 'MemPAE-ws-l2-d64-lr0.001',

        ##################################################################################


    ]

    results = collect_results()
    df_all, dfs = convert_results_to_csv(results, save_csv=False)
    pivots = make_pivots(dfs, save_csv=False)

    for base in keys:
        render(pivots, data, models, my_models, base, 
               add_avg_rank=True, use_rank=False, use_std=True, 
               use_baseline_pr=False, is_temp_tune=False, is_sort=False, is_plot=False)

    models = [
        'IForest', 'LOF', 'OCSVM', 'ECOD', 'KNN', 'PCA',  # KNN: 0.6918, LOF: 0.6612
        # 'AutoEncoder', 
        'DeepSVDD', 'GOAD', 
        'NeuTraL', 'ICL', 'MCM', 'DRL',
        'Disent',
    ]
    my_models = [
        # 'PAE-ws-pos_query+tokesn-d64-lr0.001', # Final architecture for PAE
        'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1', # 0.6878    3.7500 (SOTA! KNN: 4.2500)
        # 'PDRL-ws-pos_query+token-d64-lr0.001',
    ]

    dataname_list = [
        # '32_shuttle', # 49k: 9 numeric
        # '5_campaign', # 41k: 
        '23_mammography', # 11k: 5 numeric and 1 binary -------------
        '30_satellite', # 6k: image 
        '31_satimage-2', # 5k: image 
        '26_optdigits', # 5k: image
        '38_thyroid', # 3k: medical: 1 catgorical and 5 numeric -------------
        '7_cardiotocography', # -------------
        '18_ionosphere', # 0.3k: frequency and purse data
        '6_cardio', # 1.8k -------------
        '29_pima', # 0.7k
        '4_breastw', # 0.6k
     
    ]

    # todo: make name shorter.
    if args.synthetic:
        
        models = [
            'IForest', 
            'LOF', 
            'OCSVM', 
            'KNN', 
            'PCA',  
            'MCM', 'DRL',
            'Disent',
        ]


        anomaly_type_list = [
            'global_anomalies_',
            'cluster_anomalies_',
            'local_anomalies_',
            'dependency_anomalies_',
        ]
        irrelevant_features_list = [
            '',  
            # 'irrelevant_features_0.1_',
            # 'irrelevant_features_0.3_',
            # 'irrelevant_features_0.5_',
        ]

        suffix = '_42'
        keys = [
            'ratio_1.0_AUCPR',
        ]
        for anomaly_type in anomaly_type_list:
            synthetic_data = []
            for dataname in dataname_list:
                for feature in irrelevant_features_list:
                    file_name = f"{anomaly_type}{feature}{dataname}{suffix}"
                    synthetic_data.append(file_name)
                

            for base in keys:
                df_render = render(pivots, synthetic_data, models, my_models, base,
                    add_avg_rank=True, use_rank=False, use_std=False, use_baseline_pr=False, 
                    use_alias=True, is_temp_tune=False, is_synthetic=True, synthetic_type=anomaly_type)
    if args.contamination:
        models=[ 
            'KNN', 
            'MCM', 
            'DRL', 
            'Disent',
        ]
        # success case: cardio, sat (maybe)
        dataname_list = [
            # we can use only pima and arrhythmia
            'pima', # good
            'arrhythmia', # good
            # 'cardio', 
            # 'cardiotocography',
            # 'breastw',
            'glass',
            'wbc', 
            'wine', 
            'campaign', 
            'ionosphere',
            # 
            # 'satimage-2',
            # 'pendigits',
            'shuttle', 
            'satellite', 
            # 'thyroid',
        ]
        contamination_ratio = [
            'contam0.01',
            'contam0.03',
            'contam0.05',
        ]
        keys = [
            'ratio_1.0_AUCPR',
        ]
        for dataname in dataname_list:
            synthetic_data = [dataname.split('_')[-1]]
            print(synthetic_data)
            for contamination in contamination_ratio:
                file_name = f"{dataname}_{contamination}"
                synthetic_data.append(file_name)

            for base in keys:
                df_render = render(pivots, synthetic_data, models, my_models, base,
                    add_avg_rank=True, use_rank=False, use_std=True, use_baseline_pr=False, 
                    use_alias=True, is_temp_tune=False, is_synthetic=True, synthetic_type=contamination)
    if args.hp_ratio:
        render_hp(pivots)
        # cardio, optdigits, 
        # cardio, optdigits, wbc

    if args.npt:
        render_ours_on_npt(pivots, )
    if args.train_ratio:
        render_train_ratio(pivots, False)
        render_train_ratio_average(pivots)
        # render_memory_analysis(pivots)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--contamination', action='store_true')
    parser.add_argument('--hp_ratio', action='store_true')
    parser.add_argument('--npt', action='store_true')
    parser.add_argument('--train_ratio', action='store_true')
    parser.add_argument('--synthetic_type', type=str, default='dependency')
    args = parser.parse_args()
    main(args)