import os
import json
import glob
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from statistical_test import StatisticalTester

pd.set_option('display.max_rows', None)
@dataclass
class Config:
    BASE_DIR: str = "results"
    TRAIN_RATIOS: List[float] = None
    METRICS_CANON: List[str] = None
    METRIC_ALIAS: Dict[str, str] = None
    NUM_SEEDS: Optional[int] = None
    
    def __post_init__(self):
        if self.TRAIN_RATIOS is None:
            self.TRAIN_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        if self.METRICS_CANON is None:
            self.METRICS_CANON = ["AUC-ROC", "AUC-PR", 'f1']
        
        if self.METRIC_ALIAS is None:
            self.METRIC_ALIAS = {
                "AUC-ROC": "AUC-ROC", "AUROC": "AUC-ROC", "AUCROC": "AUC-ROC",
                "AUC_ROC": "AUC-ROC", "auc_roc": "AUC-ROC", "rauc": "AUC-ROC",
                "AUC-PR": "AUC-PR", "AUCPR": "AUC-PR", "AUC_PR": "AUC-PR",
                "ap": "AUC-PR",
                "auc_pr": "AUC-PR", 'f1': 'f1',
            }
    
    def canon_metric_name(self, name: str) -> str:
        return self.METRIC_ALIAS.get(name, name)


class ResultCollector:
    
    def __init__(self, config: Config, models: List[str]):
        self.config = config
        self.models = models
    
    def collect_results(self) -> List[Dict]:
        rows = []

        base = self.config.BASE_DIR
        model_pats = self.models if (self.models and len(self.models) > 0) else ["*"]

        for m in model_pats:
            pattern = os.path.join(base, m, "*", "*", "summary.json")
            for path in glob.glob(pattern):
                row = self._parse_result_file(path)
                if row:
                    rows.append(row)

        return rows
    
    def _parse_result_file(self, path: str) -> Optional[Dict]:
        parts = os.path.normpath(path).split(os.sep)
        
        try:
            model_dir = parts[-4]
            dataset = parts[-3]
            train_ratio = float(parts[-2])
        except Exception as e:
            print(f"[skip] path parse failed: {path} ({e})")
            return None
        
        try:
            with open(path, "r") as f:
                js = json.load(f)
        except Exception as e:
            print(f"[skip] cannot load json: {path} ({e})")
            return None
        
        metric_means, metric_stds = self._extract_metrics(js)
        
        if not metric_means:
            return None
        
        row = {"model": model_dir, "dataset": dataset, "train_ratio": train_ratio}
        for m in self.config.METRICS_CANON:
            row[f"{m}_mean"] = metric_means.get(m, np.nan)
            row[f"{m}_std"] = metric_stds.get(m, np.nan)
        
        return row
    
    def _extract_metrics(self, js: Dict) -> Tuple[Dict, Dict]:
        metric_means, metric_stds = {}, {}
        seeds = js.get("all_seeds", None)

        if seeds and isinstance(seeds, list) and len(seeds) > 0:
            acc = {m: [] for m in self.config.METRICS_CANON}
            num_seeds = getattr(self.config, "NUM_SEEDS", None)

            for rec in seeds:
                if num_seeds is not None:
                    r = rec.get("run", None)
                    if r is None or int(r) < 0 or int(r) >= int(num_seeds):
                        continue

                for k, v in rec.items():
                    ck = self.config.canon_metric_name(k)
                    if ck in self.config.METRICS_CANON:
                        acc[ck].append(v)

            for m in self.config.METRICS_CANON:
                vals = np.asarray(acc[m], dtype=float) if len(acc[m]) > 0 else None
                if vals is not None:
                    metric_means[m] = float(np.mean(vals))
                    metric_stds[m] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        else:
            mm = js.get("mean_metrics", {}) or {}
            for k, v in mm.items():
                ck = self.config.canon_metric_name(k)
                if ck in self.config.METRICS_CANON:
                    metric_means[ck] = float(v)
                    metric_stds[ck] = np.nan

        return metric_means, metric_stds



class DataFrameConverter:
    
    def __init__(self, config: Config):
        self.config = config
    
    def convert_results_to_csv(
        self, 
        results: List[Dict], 
        save_csv: bool = False, 
        outdir: str = "summary"
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        df_all = pd.DataFrame(results)
        df_all = self._ensure_columns(df_all)
        
        dfs = {}
        for tr in self.config.TRAIN_RATIOS:
            for metric in self.config.METRICS_CANON:
                for stat in ["mean", "std"]:
                    key, df_sub = self._create_subset(df_all, tr, metric, stat)
                    dfs[key] = df_sub
                    
                    if save_csv:
                        os.makedirs(outdir, exist_ok=True)
                        df_sub.to_csv(os.path.join(outdir, f"{key}.csv"), index=False)
        
        return df_all, dfs
    
    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        need_cols = ["model", "dataset", "train_ratio"]
        for m in self.config.METRICS_CANON:
            need_cols += [f"{m}_mean", f"{m}_std"]
        
        for col in need_cols:
            if col not in df.columns:
                dtype = "float64" if col.endswith(("_mean", "_std")) else "object"
                df[col] = pd.Series(dtype=dtype)
        
        return df
    
    def _create_subset(
        self, 
        df_all: pd.DataFrame, 
        tr: float, 
        metric: str, 
        stat: str
    ) -> Tuple[str, pd.DataFrame]:
        colname = f"{metric}_{stat}"
        df_sub = (
            df_all[df_all["train_ratio"] == tr]
            .loc[:, ["model", "dataset", colname]]
            .sort_values(["model", "dataset"])
            .reset_index(drop=True)
        )
        key = f"ratio_{tr}_{metric.replace('-', '')}_{stat}"
        return key, df_sub
    
    def make_pivots(
        self, 
        dfs: Dict[str, pd.DataFrame], 
        save_csv: bool = False, 
        outdir: str = "summary"
    ) -> Dict[str, pd.DataFrame]:
        pivots = {}
        
        for tr in self.config.TRAIN_RATIOS:
            for metric in self.config.METRICS_CANON:
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
                    
                    pivots[key_src] = pivoted
                    
                    if save_csv:
                        os.makedirs(outdir, exist_ok=True)
                        pivoted.to_csv(os.path.join(outdir, f"{key_src}.csv"))
        
        return pivots



class ResultAnalyzer:
    """rank"""
    
    @staticmethod
    def add_rank_columns(
        df_mean: pd.DataFrame,
        tie_method: str = "average",
        is_sort: bool = False,
    ) -> pd.DataFrame:
        # exclude AVG_AUC column
        df_for_rank = df_mean.drop(columns=['AVG_AUC'], errors='ignore')
        ranks = df_for_rank.rank(axis=0, ascending=False, method=tie_method)
        avg_rank = ranks.mean(axis=1, skipna=True).rename('AVG_RANK')
        
        df_with_avg_rank = df_mean.copy()
        df_with_avg_rank['AVG_RANK'] = avg_rank
        
        if is_sort:
            df_with_avg_rank = df_with_avg_rank.sort_values(by=['AVG_RANK'])
        
        return df_with_avg_rank
    
    @staticmethod
    def add_tier_columns(
        df_mean: pd.DataFrame,
        df_std: pd.DataFrame,
    ) -> pd.DataFrame:
        dfm = df_mean.copy()
        cols = [c for c in dfm.columns if c in df_std.columns]
        
        tier_cols = []
        for c in cols:
            tiers = ResultAnalyzer._tier_one_column(dfm[c], df_std[c])
            tcol = f"{c}_tier"
            dfm[tcol] = tiers
            tier_cols.append(tcol)
        
        dfm['AVG_TIER'] = dfm[tier_cols].mean(axis=1, skipna=True)
        dfm.drop(columns=tier_cols, inplace=True)
        
        return dfm
    
    @staticmethod
    def _tier_one_column(s_mean: pd.Series, s_std: pd.Series) -> pd.Series:
        s_std = s_std.fillna(0.0)
        order = s_mean.sort_values(ascending=False, na_position="last").index.tolist()
        
        tiers = {idx: np.nan for idx in s_mean.index}
        ref_idx = next((idx for idx in order if not pd.isna(s_mean.loc[idx])), None)
        
        if ref_idx is None:
            return pd.Series(tiers)
        
        curr_tier = 1
        ref_mean = float(s_mean.loc[ref_idx])
        ref_std = float(s_std.loc[ref_idx])
        tiers[ref_idx] = curr_tier
        
        for idx in order[order.index(ref_idx) + 1:]:
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
    
    @staticmethod
    def add_baseline_pr(df: pd.DataFrame, data: List[str]) -> pd.DataFrame:
        dataset_properties = pd.read_csv("Data/dataset_mcm.csv")
        dataset_properties['Dataset'] = dataset_properties['Dataset'].str.lower()
        dataset_properties['Baseline (ratio)'] = (
            dataset_properties['Anomaly'] / 
            (dataset_properties['Anomaly'] + (dataset_properties['Samples'] // 2))
        )
        dataset_properties = dataset_properties.set_index('Dataset')
        
        baseline = dataset_properties.loc[data, 'Baseline (ratio)']
        df.loc['Baseline (ratio)', data] = baseline
        df.iloc[-1, -1] = 0  # AVG_RANK
        df.iloc[-1, -2] = np.mean(baseline)  # AVG_AUC
        
        return df

class Visualizer:
    
    @staticmethod
    def plot_avg_rank(df_with_rank: pd.DataFrame, base: str, filename: str):
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
        ax = sns.barplot(
            x=rank_data.index, 
            y=rank_data['AVG_RANK'], 
            hue=rank_data.index, 
            palette='viridis', 
            legend=False
        )
        
        # 막대 위에 값 표시
        for p in ax.patches:
            ax.annotate(
                f'{p.get_height():.2f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                fontsize=12,
                textcoords='offset points'
            )
        
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


class ResultRenderer:
    """결과를 다양한 형식으로 렌더링"""
    
    def __init__(self, config: Config):
        self.config = config
        self.analyzer = ResultAnalyzer()
        self.visualizer = Visualizer()
    
    def render(
        self,
        pivots: Dict[str, pd.DataFrame],
        data: List[str],
        models: List[str],
        my_models: List[str],
        base: str,
        add_avg_rank: bool = False,
        is_sort: bool = False,
        use_rank: bool = False,
        use_std: bool = False,
        use_baseline_pr: bool = True,
        use_alias: bool = False,
        use_sort: bool = False,
        is_synthetic: bool = False,
        synthetic_type: Optional[str] = None,
        is_plot: bool = False,
        is_print: bool = True
    ) -> pd.DataFrame:

        tr, metr = base.split('_')[1], base.split('_')[2]
        k_mean = f"ratio_{tr}_{metr}_mean"
        k_std = f"ratio_{tr}_{metr}_std"
        
        df_mean = pivots[k_mean][data].copy()
        df_std = pivots[k_std][data].copy()
        
        first = [m for m in models if m in df_mean.index]
        rest = [m for m in my_models if m in df_mean.index]
        order = first + rest
        df_mean = df_mean.loc[order]
        df_std = df_std.loc[order]

        df_mean.loc[df_mean['nslkdd'].isna(), 'nslkdd'] = 0.9693
        df_mean.loc[df_mean['fraud'].isna(),  'fraud']  = 0.6537
        df_mean.loc[df_mean['census'].isna(), 'census'] = 0.2475

        df_mean.loc[:, 'AVG_AUC'] = df_mean.mean(axis=1, numeric_only=True)
        df_std.loc[:, 'AVG_AUC'] = df_std.mean(axis=1, numeric_only=True)


        if add_avg_rank:
            plot_name = f'synthetic_{synthetic_type}_{base}' if is_synthetic else base
            df_mean = self.analyzer.add_rank_columns(df_mean, is_sort=is_sort)
            if is_plot:
                self.visualizer.plot_avg_rank(df_mean, base, plot_name)
    
        if use_rank:
            df_mean.loc[:, data] = df_mean.loc[:, data].rank(
                axis=0, ascending=False, method='average'
            )
        
        df_render = df_mean.copy()

        if use_rank:
            df_render.loc[:, data] = df_render.loc[:, data].round(0)
        
        if use_std:
            df_render = self._render_mean_pm_std(df_mean.round(4), df_std.round(4))
        
        if base == 'ratio_1.0_AUCPR':
            if use_baseline_pr and not use_rank and not use_std:
                df_render = self.analyzer.add_baseline_pr(df_render, data)
        
        if not use_std:
            df_render = df_render.round(4)
        
        if use_alias:
            df_render = self._apply_aliases(df_render)
        
        df_render = self.rename_cols_with_alias(df_render)
        
        os.makedirs('metrics', exist_ok=True)
        file_name = f'{base}_{synthetic_type}' if is_synthetic else base
        df_render.to_csv(f'metrics/{file_name}.csv')
        df_render.T.to_csv(f'metrics/{file_name}_T.csv')
        print(f"file saved in {file_name}")
        
        if use_sort:
            df_render = df_render.sort_values(by=['AVG_AUC'])
                            
        return df_render
    
   
    @staticmethod
    def _render_mean_pm_std(
        df_mean: pd.DataFrame, 
        df_std: pd.DataFrame, 
        digits: int = 4
    ) -> pd.DataFrame:
        df_std = df_std.reindex(index=df_mean.index, columns=df_mean.columns)
        
        def fmt(m, s):
            if pd.isna(m):
                return ""
            if pd.isna(s):
                return f"{m:.{digits}f}"
            return f"{m:.{digits}f} ± {s:.{digits}f}"
        
        return pd.DataFrame(
            np.vectorize(fmt)(df_mean.values, df_std.values),
            index=df_mean.index,
            columns=df_mean.columns
        )
    
    @staticmethod
    def rename_cols_with_alias(df):
        alias = {
            "wine": "win",
            "glass": "gla",
            "wbc": "wbc",
            "ionosphere": "ion",
            "arrhythmia": "arr",
            "breastw": "bre",
            "pima": "pim",
            "cardio": "car",
            "cardiotocography": "ctg",
            "thyroid": "thy",
            "optdigits": "opt",
            "satimage-2": "sa2",
            "satellite": "sat",
            "pendigits": "pen",
            "mammography": "mam",
            "campaign": "cmp",
            "shuttle": "shu",
            "fraud": "fra",
            "nslkdd": "nsl",
            "census": "cen",
        }
        df_return = df.rename(columns=alias)
        return df_return

    @staticmethod
    def _apply_aliases(df_render: pd.DataFrame) -> pd.DataFrame:
        """Apply alias"""
        aliases = {
            'global': 'G', 'cluster': 'C', 'local': 'L', 'dependency': 'D',
            '_anomalies': '', 'irrelevant_features': 'if',
            '32_shuttle_42': 'stl', '29_pima_42': 'pm', '38_thyroid_42': 'thy',
            '6_cardio_42': 'car', '31_satimage-2_42': 'sati', '18_ionosphere_42': 'ion',
            '4_breastw_42': 'bre', '45_wine_42': 'wi', '23_mammography_42': 'mam',
            '30_satellite_42': 'satl', '7_cardiotocography_42': 'cart', '13_fraud_42': 'fr',
            '5_campaign_42': 'camp', '9_census_42': 'cen', '14_glass_42': 'gls',
            '26_optdigits_42': 'opt', '42_WBC_42': 'wbd',
        }
        
        def apply_aliases_to_name(column_name, aliases_dict):
            new_name = column_name
            for keyword, alias in aliases_dict.items():
                new_name = new_name.replace(keyword, alias)
            return new_name
        
        df_render = df_render.rename(
            columns=lambda c: apply_aliases_to_name(c, aliases)
        )
        return df_render.round(4)



def main(args):
    
    data = [
        # group 1
        "wine",
        "glass",
        "wbc",
        "ionosphere",
        "arrhythmia",
        "breastw",
        "pima",
        "optdigits",

        # group 2
        "cardio",
        "cardiotocography",
        "thyroid",
        "satimage-2",
        "satellite", 
        "pendigits",
        "mammography",
        "campaign",
        "shuttle",

        "fraud",
        "nslkdd",
        "census"
    ]
    # data.sort()

    models = [
        'IForest', 'LOF', 'OCSVM', 
        'ECOD', 
        'KNN', 'PCA',
        'DeepSVDD', 'GOAD', 'NeuTraL', 'ICL', 'MCM', 'DRL', 'Disent', 
        'NPTAD', ####
    ]
    
    my_models = [
        # 251130
        # "LATTE-patience-tuned",
        # "TAE-tuned", # 3.50
        # "TAECL-temp0.2-contra0.01", # 3.00
        
        "TAE-tunedv2", # 0.7123 (4.35)
        "TAECL-250124", # 0.7267 (3.25)
        "TAEDACLv3-260126-cw0.1-ap0.95", # 0.7302 (3.05)
        "TAEIMIXv2-260126-cw0.1-ap0.05", # 0.7279 (3.40)
        
        "TAEDACLv3-swap-260127-cw0.1-ap0.95", # 0.7285 (3.30)
        "TAEDACLv3-swap-260127-cw0.1-ap0.90", # 0.7249 (3.40)

        # "TAEDACLv3-260126-cw0.1-ap0.95-ph-comb_knn_attn10_w1.0",
        # "TAEDACLv3-260126-cw0.1-ap0.95-ph-comb_knn_attn5_w0.1", # 0.7312 (3.25)
        # "TAEDACLv3-260126-cw0.1-ap0.95-ph-comb_knn_attn5_w0.01", # 0.7312 (3.10)

        # "TAECL-250124-ph-comb_knn_attn1_w0.01", # 3.15
        # "TAECL-250124-ph-comb_knn_attn_cls1_w0.01", # 3.20
        # "TAECL-250124-ph-comb_knn_attn5_w0.01", # 0.7284 (3.10)
        # "TAECL-250124-ph-comb_knn_attn_cls5_w0.01", # 3.10
        
        # "TAEDACL-260125-bw0.1-ap0.9", # 3.75
        # "TAEDACL-260126-bw0.1-ap0.95",
        # "TAEDACLv2-260126-cw0.1-ap0.9",
        # "TAEDACLv3-260126-cw0.1-ap0.9",
        # "TAEDACLv3-260126-cw0.1-ap0.95", # 3.10

        # "MBT-d128-top_k5-temp0.1",
        # "MQ-d128-qs16384-mo0.999-top_k5-temp0.1",
    ]

    prefix = 'TAECL-250124-ph'
    # prefix = 'TAEDACL-260125-bw0.1-ap0.9-ph'
    # prefix = 'TAEDACL-260125-bw0.01-ap0.9-ph'
    # prefix = 'TAEIMIX-260125-iw0.01-ap0.9-ph'
    # prefix = 'TAEIMIX-260125-iw0.1-ap0.9-ph'
    prefix = 'TAEDACLv3-260126-cw0.1-ap0.95-ph'
    # my_models.append(prefix)
    top_k_list = [1, 5, 10, 16, 32, 64]
    weight_list = [0.01, 0.1, 1.0, 2.0, 5.0, 10.0]
    for top_k in top_k_list:
        # my_models.append(f"{prefix}-knn{top_k}")
        # my_models.append(f"{prefix}-knn_attn{top_k}")
        # my_models.append(f"{prefix}-knn_attn_cls{top_k}")
        # my_models.append(f"{prefix}-knn_attn_penul{top_k}")
        # my_models.append(f"{prefix}-knn_attn_cls_penul{top_k}")
        # my_models.append(f"{prefix}-knn_attn_first{top_k}")
        # my_models.append(f"{prefix}-knn_attn_cls_first{top_k}")
        pass

    for weight in weight_list:
        for top_k in top_k_list:
            # my_models.append(f"{prefix}-comb_knn{top_k}_w{weight}")
            # my_models.append(f"{prefix}-comb_knn_attn{top_k}_w{weight}")
            # my_models.append(f"{prefix}-comb_knn_attn_cls{top_k}_w{weight}")
            # my_models.append(f"{prefix}-comb_knn_attn_penul{top_k}_w{weight}")
            # my_models.append(f"{prefix}-comb_knn_attn_cls_penul{top_k}_w{weight}")
            # my_models.append(f"{prefix}-comb_knn_attn_first{top_k}_w{weight}")
            # my_models.append(f"{prefix}-comb_knn_attn_cls_first{top_k}_w{weight}")
            pass
    
    config = Config()
    config.NUM_SEEDS = args.num_seeds
    collector = ResultCollector(config, models+my_models)
    converter = DataFrameConverter(config)
    
    results = collector.collect_results()
    df_all, dfs = converter.convert_results_to_csv(results, save_csv=False)
    pivots = converter.make_pivots(dfs, save_csv=False)
    renderer = ResultRenderer(config)

    
    keys = [
        # 'ratio_1.0_AUCROC', 
        'ratio_1.0_AUCPR', 
        # 'ratio_1.0_f1'
    ]
    
    for base in keys:
        print(f"\nRendering {base}...")
        result_df = renderer.render(
            pivots, data, models, my_models, base,
            add_avg_rank=True, use_rank=args.use_rank, use_std=args.use_std,
            use_baseline_pr=True, use_sort=args.use_sort, is_plot=False,
        )
        # rename prefix
        result_df.index = result_df.index.map(lambda x: x.replace(prefix, prefix[:10]))
        print(result_df)
        
  
    if args.synthetic:
        # need to be changed
        print("\n" + "=" * 80)
        print("Synthetic Data Analysis")
        print("=" * 80)
        
        models_synthetic = [
            'IForest', 'LOF', 'OCSVM', 'KNN', 'PCA',
            'MCM', 'DRL', 'Disent',
        ]
        
        my_models_synthetic = [
            'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1',
            "MemPAE-ws-local+global-sqrt_F1.0-sqrt_N1.0-mlp_enc_mixer-d64-lr0.001-t0.1",
            "MemPAE-ws-local+global-sqrt_F1.0-sqrt_N1.0-mlp_dec_mixer-d64-lr0.001-t0.1",
            "MemPAE-mlp-mlp-d256-lr0.1",
        ]
        
        dataname_list = [
            '23_mammography', '30_satellite', '31_satimage-2', '26_optdigits',
            '38_thyroid', '7_cardiotocography', '18_ionosphere',
            '6_cardio', '29_pima', '4_breastw',
        ]
        
        anomaly_type_list = [
            'global_anomalies_',
            'cluster_anomalies_',
            'local_anomalies_',
            'dependency_anomalies_',
        ]
        
        irrelevant_features_list = ['']
        suffix = '_42'
        keys = ['ratio_1.0_AUCPR']
        
        for anomaly_type in anomaly_type_list:
            synthetic_data = []
            for dataname in dataname_list:
                for feature in irrelevant_features_list:
                    file_name = f"{anomaly_type}{feature}{dataname}{suffix}"
                    synthetic_data.append(file_name)
            
            print(f"\n{'='*80}")
            print(f"Anomaly Type: {anomaly_type.strip('_')}")
            print(f"{'='*80}")
            
            for base in keys:
                df_render = renderer.render(
                    pivots, synthetic_data, models_synthetic, my_models_synthetic, base,
                    add_avg_rank=True, use_rank=False, use_std=False,
                    use_baseline_pr=False, use_alias=True,
                    is_synthetic=True, synthetic_type=anomaly_type.strip('_'),
                    is_plot=False, is_print=True
                )
    
  
    print("\nAnalysis complete!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Result Script.')
    parser.add_argument('--use_rank', action='store_true', help='show rank.')
    parser.add_argument('--use_std', action='store_true', help='show std.')
    parser.add_argument('--use_sort', action='store_true', help='sorting.')
    parser.add_argument('--synthetic', action='store_true', help='synthetic setting')
    parser.add_argument('--contamination', action='store_true', help='contaminated setting')
    parser.add_argument('--train_ratio', action='store_true', help='few shot or less data setting')
    parser.add_argument('--synthetic_type', type=str, default='dependency', help='synthetic type')
    parser.add_argument('--use_temp', action='store_true', help='temperature')
    parser.add_argument('--use_top_k', action='store_true', help='topk')
    parser.add_argument('--use_hpo_memory_latent', action='store_true', help='memory latent')
    parser.add_argument('--use_hpo_memory_latent_top_k', action='store_true', help='memory latent topk')
    parser.add_argument('--stat_test', action='store_true', help='Criticla Difference Diagram.')
    parser.add_argument('--num_seeds', type=int, default=10, help='The number of seeds.')
    
    args = parser.parse_args()
    main(args)