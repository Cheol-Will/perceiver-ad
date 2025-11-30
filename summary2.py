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
    """실험 설정을 관리하는 클래스"""
    BASE_DIR: str = "results"
    TRAIN_RATIOS: List[float] = None
    METRICS_CANON: List[str] = None
    METRIC_ALIAS: Dict[str, str] = None
    
    def __post_init__(self):
        if self.TRAIN_RATIOS is None:
            self.TRAIN_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        if self.METRICS_CANON is None:
            self.METRICS_CANON = ["AUC-ROC", "AUC-PR", 'f1']
        
        if self.METRIC_ALIAS is None:
            self.METRIC_ALIAS = {
                "AUC-ROC": "AUC-ROC", "AUROC": "AUC-ROC", "AUCROC": "AUC-ROC",
                "AUC_ROC": "AUC-ROC", "auc_roc": "AUC-ROC",
                "AUC-PR": "AUC-PR", "AUCPR": "AUC-PR", "AUC_PR": "AUC-PR",
                "auc_pr": "AUC-PR", 'f1': 'f1'
            }
    
    def canon_metric_name(self, name: str) -> str:
        """메트릭 이름을 표준 형식으로 변환"""
        return self.METRIC_ALIAS.get(name, name)


# ============================================================================
# 데이터 수집 및 처리 클래스
# ============================================================================

class ResultCollector:
    """실험 결과를 수집하고 처리하는 클래스"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def collect_results(self) -> List[Dict]:
        """results 디렉토리에서 모든 summary.json 파일을 수집"""
        rows = []
        pattern = os.path.join(self.config.BASE_DIR, "*", "*", "*", "summary.json")
        
        for path in glob.glob(pattern):
            row = self._parse_result_file(path)
            if row:
                rows.append(row)
        
        return rows
    
    def _parse_result_file(self, path: str) -> Optional[Dict]:
        """개별 결과 파일을 파싱"""
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
        """JSON에서 메트릭 평균과 표준편차를 추출"""
        metric_means, metric_stds = {}, {}
        seeds = js.get("all_seeds", None)
        
        if seeds and isinstance(seeds, list) and len(seeds) > 0:
            # 여러 시드 결과가 있는 경우
            acc = {m: [] for m in self.config.METRICS_CANON}
            for rec in seeds:
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
            # 평균값만 제공된 경우
            mm = js.get("mean_metrics", {}) or {}
            for k, v in mm.items():
                ck = self.config.canon_metric_name(k)
                if ck in self.config.METRICS_CANON:
                    metric_means[ck] = float(v)
                    metric_stds[ck] = np.nan
        
        return metric_means, metric_stds


# ============================================================================
# 데이터 변환 클래스
# ============================================================================

class DataFrameConverter:
    """수집된 결과를 다양한 형태의 DataFrame으로 변환"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def convert_results_to_csv(
        self, 
        results: List[Dict], 
        save_csv: bool = False, 
        outdir: str = "summary"
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """결과를 DataFrame으로 변환하고 선택적으로 CSV 저장"""
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
        """필요한 컬럼이 모두 존재하도록 보장"""
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
        """특정 train_ratio, metric, stat에 대한 부분집합 생성"""
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
        """데이터를 피벗 테이블로 변환 (model x dataset)"""
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


# ============================================================================
# 분석 클래스
# ============================================================================

class ResultAnalyzer:
    """결과 분석 및 랭킹 계산"""
    
    @staticmethod
    def add_rank_columns(
        df_mean: pd.DataFrame,
        tie_method: str = "average",
        is_sort: bool = False,
    ) -> pd.DataFrame:
        """각 데이터셋에서의 랭킹과 평균 랭킹 계산"""
        ranks = df_mean.rank(axis=0, ascending=False, method=tie_method)
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
        """표준편차를 고려한 티어 분류 추가"""
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
        """단일 컬럼에 대한 티어 계산"""
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
        """Baseline (ratio) 행 추가 (AUC-PR 전용)"""
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


# ============================================================================
# 시각화 클래스
# ============================================================================

class Visualizer:
    """결과 시각화"""
    
    @staticmethod
    def plot_avg_rank(df_with_rank: pd.DataFrame, base: str, filename: str):
        """평균 랭킹 바 플롯 생성"""
        rank_data = df_with_rank[['AVG_RANK']].sort_values(by='AVG_RANK', ascending=False)
        
        # 모델 이름 단순화
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


# ============================================================================
# 렌더링 클래스
# ============================================================================

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
        is_synthetic: bool = False,
        synthetic_type: Optional[str] = None,
        is_plot: bool = False,
        is_print: bool = True,
        use_temp: bool = False,
        use_top_k: bool = False,
        use_hpo_memory_latent: bool = False,
        use_hpo_memory_latent_top_k: bool = False,
    ) -> pd.DataFrame:
        """
        주요 렌더링 함수: 피벗 데이터를 포맷팅하고 시각화
        
        Parameters:
        -----------
        pivots : 피벗 테이블 딕셔너리
        data : 포함할 데이터셋 리스트
        models : 베이스라인 모델 리스트
        my_models : 제안 모델 리스트
        base : 메트릭 키 (예: 'ratio_1.0_AUCPR')
        ... (기타 옵션들)
        
        Returns:
        --------
        pd.DataFrame : 렌더링된 결과 DataFrame
        """
        tr, metr = base.split('_')[1], base.split('_')[2]
        k_mean = f"ratio_{tr}_{metr}_mean"
        k_std = f"ratio_{tr}_{metr}_std"
        
        df_mean = pivots[k_mean][data].copy()
        df_std = pivots[k_std][data].copy()
        
        # HPO 관련 처리 (필터링 전에 수행)
        if use_hpo_memory_latent_top_k:
            df_mean = self._apply_hpo_memory_latent_top_k(df_mean)
        
        if use_top_k:
            df_mean = self._apply_top_k(df_mean)
        
        if use_hpo_memory_latent:
            df_mean = self._apply_hpo_memory_latent(df_mean)

        if use_temp:
            df_mean = self._apply_temp(df_mean)
            
        
        # 모델 순서 재배열 (HPO 처리 이후)
        first = [m for m in models if m in df_mean.index]
        rest = [m for m in my_models if m in df_mean.index]  # 존재하는 모델만 선택
        order = first + rest
        df_mean = df_mean.loc[order]
        df_std = df_std.loc[order]
        
        # df_mean.loc['NPTAD', 'census'] = 0.2672
        # df_mean.loc['NPTAD', 'fraud'] = 0.3868
        # df_mean.loc['LATTE-Full_rank-no_dec-d64-lr0.001-g0.99-t0.1-p20', 'fraud'] = 0.5100
        # df_mean.loc['LATTE-patience', :] = df_mean.loc[latte_patience, :].max(axis=0, numeric_only=True)
        # df_mean.loc['LATTE-patience', 'census'] = 0.2474
        # df_mean.drop(latte_patience, axis=0, inplace=True)
        # df_mean.loc["LATTE-patience-tuned", 'census'] = 0.2474

        df_mean.loc[:, 'AVG_AUC'] = df_mean.mean(axis=1, numeric_only=True)
        df_std.loc[:, 'AVG_AUC'] = df_std.mean(axis=1, numeric_only=True)



        # 랭킹 추가
        if add_avg_rank:
            plot_name = f'synthetic_{synthetic_type}_{base}' if is_synthetic else base
            df_mean = self.analyzer.add_rank_columns(df_mean, is_sort=is_sort)
            if is_plot:
                self.visualizer.plot_avg_rank(df_mean, base, plot_name)
        
        # 랭킹 표시 모드
        if use_rank:
            df_mean.loc[:, data] = df_mean.loc[:, data].rank(
                axis=0, ascending=False, method='average'
            )
        
        df_render = df_mean.copy()
        
        if use_rank:
            df_render.loc[:, data] = df_render.loc[:, data].round(0)
        
        if use_std:
            df_render = self._render_mean_pm_std(df_mean.round(4), df_std.round(4))
        
        # Baseline 추가 (AUC-PR 전용)
        if base == 'ratio_1.0_AUCPR':
            if use_baseline_pr and not use_rank and not use_std:
                df_render = self.analyzer.add_baseline_pr(df_render, data)
        
        if not use_std:
            df_render = df_render.round(4)
        
        # 별칭 적용
        if use_alias:
            df_render = self._apply_aliases(df_render)
        
        # 모델 이름 정리
        df_render.index = [
            c.replace("pos_query+token", "pos") if "pos_query+token" in c else c
            for c in df_render.index
        ]
        df_render.index = [
            c.replace("MemPAE-ws-cross_attn", "MPCA") if "MemPAE-ws-cross_attn" in c else c
            for c in df_render.index
        ]
        
        # 저장
        os.makedirs('metrics', exist_ok=True)
        file_name = f'{base}_{synthetic_type}' if is_synthetic else base
        df_render.to_csv(f'metrics/{file_name}.csv')
        df_render.T.to_csv(f'metrics/{file_name}_T.csv')
        print(f"file saved in {file_name}")
        
        if is_print:
            print(base)
            print(df_render)
            print()
        
        return df_render
    
    def _apply_hpo_memory_latent_top_k(self, df_mean: pd.DataFrame) -> pd.DataFrame:
        """HPO 메모리/레이턴트 + top-k 적용"""
        our_model = ['MemPAE-ws-pos_query+token-d64-lr0.001-t0.1']
        
        # top-k 없는 모델들
        num_latent_list = [0.5, 1.0, 2.0, 4.0]
        num_memory_list = [0.5, 1.0, 2.0, 4.0]
        temperature_list = [0.1]
        
        for l in num_latent_list:
            for m in num_memory_list:
                for t in temperature_list:
                    model_name = f"MemPAE-ws-local+global-sqrt_F{l}-sqrt_N{m}-d64-lr0.001-t{t}"
                    our_model.append(model_name)
        
        # top-k 있는 모델들
        top_k_list = [1, 5, 10, 15]
        temperature_list = [0.1, 0.5, 1.0]
        
        for k in top_k_list:
            for l in num_latent_list:
                for m in num_memory_list:
                    for t in temperature_list:
                        model_name = f"MemPAE-ws-local+global-sqrt_F{l}-sqrt_N{m}-top{k}-d64-lr0.001-t{t}"
                        our_model.append(model_name)
        
        print("\n" + "="*80)
        print("Best Model per Dataset (HPO Memory Latent)")
        print("="*80)
        for col in df_mean.columns:
            max_idx = df_mean.loc[our_model, col].idxmax()
            max_val = df_mean.loc[our_model, col].max()
            print(f"{col:20s} -> {max_idx:70s} (score: {max_val:.4f})")
        print("="*80 + "\n")


        # 최대값으로 통합
        df_mean.loc[our_model[0], :] = df_mean.loc[our_model, :].max(axis=0)
        df_mean.drop(our_model[1:], axis=0, inplace=True)
        
        return df_mean
    
    def _apply_temp(self, df_mean: pd.DataFrame) -> pd.DataFrame:
        """top-k 모델 선택"""
        our_model = ['MemPAE-ws-pos_query+token-d64-lr0.001-t0.1']
        
        # for t in [0.1, 0.5, 1.0]:
        for t in [0.01, 0.05, 0.1, 0.5, 1.0]:
            model_name = f"MemPAE-ws-local+global-sqrt_F1.0-sqrt_N1.0-d64-lr0.001-t{t}"
            our_model.append(model_name)
        
        print("\n" + "="*80)
        print("Best Model per Dataset (HPO: Temperature)")
        print("="*80)
        for col in df_mean.columns:
            max_idx = df_mean.loc[our_model, col].idxmax()
            max_val = df_mean.loc[our_model, col].max()
            print(f"{col:20s} -> {max_idx:70s} (score: {max_val:.4f})")
        print("="*80 + "\n")

        df_mean.loc[our_model[0], :] = df_mean.loc[our_model, :].max(axis=0)
        df_mean.drop(our_model[1:], axis=0, inplace=True)
        
        return df_mean
    def _apply_top_k(self, df_mean: pd.DataFrame) -> pd.DataFrame:
        """top-k 모델 선택"""
        our_model = ['MemPAE-ws-pos_query+token-d64-lr0.001-t0.1']
        
        for k in [1, 5, 10, 15]:
            for t in [0.01, 0.05, 0.1, 0.5, 1.0]:
                model_name = f"MemPAE-ws-local+global-sqrt_F-sqrt_N-top{k}-d64-lr0.001-t{t}"
                our_model.append(model_name)
        
        print("\n" + "="*80)
        print("Best Model per Dataset (HPO Memory Latent)")
        print("="*80)
        for col in df_mean.columns:
            max_idx = df_mean.loc[our_model, col].idxmax()
            max_val = df_mean.loc[our_model, col].max()
            print(f"{col:20s} -> {max_idx:70s} (score: {max_val:.4f})")
        print("="*80 + "\n")

        df_mean.loc[our_model[0], :] = df_mean.loc[our_model, :].max(axis=0)
        df_mean.drop(our_model[1:], axis=0, inplace=True)
        
        return df_mean
    
    def _apply_hpo_memory_latent(self, df_mean: pd.DataFrame) -> pd.DataFrame:
        """HPO 메모리/레이턴트 적용"""
        print(f"use_hpo_memory_latent=True")
        our_model = ['MemPAE-ws-pos_query+token-d64-lr0.001-t0.1']
        
        num_latent_list = [0.5, 1.0, 2.0, 4.0]
        num_memory_list = [0.5, 1.0, 2.0, 4.0]
        temperature_list = [0.1, 0.5, 1.0]
        
        for l in num_latent_list:
            for m in num_memory_list:
                for t in temperature_list:
                    model_name = f"MemPAE-ws-local+global-sqrt_F{l}-sqrt_N{m}-d64-lr0.001-t{t}"
                    our_model.append(model_name)
        a = [
            'MemPAE-ws-pos_query-d16-lr0.001-t0.1',
            'MemPAE-ws-pos_query-d16-lr0.005-t0.1',
            'MemPAE-ws-pos_query-d16-lr0.01-t0.1',
            'MemPAE-ws-pos_query-d16-lr0.05-t0.1',
            'MemPAE-ws-pos_query-d32-lr0.001-t0.1',
            'MemPAE-ws-pos_query-d32-lr0.005-t0.1',
            'MemPAE-ws-pos_query-d32-lr0.01-t0.1',
            'MemPAE-ws-pos_query-d32-lr0.05-t0.1',
            'MemPAE-ws-pos_query-L6-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query-L2-d64-lr0.001-t0.1',

            'MemPAE-ws-pos_query+token-d64-lr0.001',
            'MemPAE-ws-pos_query+token-d64-lr0.001-t0.01',
            'MemPAE-ws-pos_query+token-d64-lr0.001-t0.05',
            'MemPAE-ws-pos_query+token-d64-lr0.001-t0.5',
            'MemPAE-ws-pos_query+token-L0-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-L2-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-L3-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-L4-d64-lr0.001-t0.2',
            'MemPAE-ws-pos_query+token-L5-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-L5-d64-lr0.005-t0.1',
            'MemPAE-ws-pos_query+token-L6-d64-lr0.001-t0.01',
            'MemPAE-ws-pos_query+token-L6-d64-lr0.001-t0.05',
            'MemPAE-ws-pos_query+token-L6-d64-lr0.001-t0.1',

            'MemPAE-ws-pos_query+token-memory_ratio0.5-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-memory_ratio2.0-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-memory_ratio4.0-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-memory_ratio8.0-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-latent_ratio0.5-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-latent_ratio2.0-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-latent_ratio4.0-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-latent_ratio8.0-d64-lr0.001-t0.1',
        ]

        our_model = our_model + a
        print("\n" + "="*80)
        print("Best Model per Dataset (HPO Memory Latent)")
        print("="*80)
        for col in df_mean.columns:
            max_idx = df_mean.loc[our_model, col].idxmax()
            max_val = df_mean.loc[our_model, col].max()
            print(f"{col:20s} -> {max_idx:70s} (score: {max_val:.4f})")
        print("="*80 + "\n")

        df_mean.loc[our_model[0], :] = df_mean.loc[our_model, :].max(axis=0)
        df_mean.drop(our_model[1:], axis=0, inplace=True)
        
        return df_mean
    
    @staticmethod
    def _render_mean_pm_std(
        df_mean: pd.DataFrame, 
        df_std: pd.DataFrame, 
        digits: int = 4
    ) -> pd.DataFrame:
        """평균 ± 표준편차 형식으로 포맷팅"""
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
    def _apply_aliases(df_render: pd.DataFrame) -> pd.DataFrame:
        """컬럼 이름에 별칭 적용"""
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


# ============================================================================
# 특수 분석 함수들
# ============================================================================

class SpecializedAnalysis:
    """특수 목적 분석 함수들"""
    
    def __init__(self, config: Config, renderer: ResultRenderer):
        self.config = config
        self.renderer = renderer
    
    def render_train_ratio(
        self, 
        pivots: Dict[str, pd.DataFrame], 
        print_summary: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """훈련 비율별 모델 성능 분석"""
        data = [
            'campaign', 'nslkdd', 'fraud', 'census',
        ]
        models = ['MCM', 'DRL', 'Disent']
        my_models = ['MemPAE-ws-pos_query+token-d64-lr0.001-t0.1']
        keys = [
            'ratio_0.2_AUCPR', 'ratio_0.4_AUCPR', 'ratio_0.6_AUCPR',
            'ratio_0.8_AUCPR', 'ratio_1.0_AUCPR',
        ]
        
        # 각 비율별 데이터프레임 수집
        dict_df = {}
        for base in keys:
            df = self.renderer.render(
                pivots, data, models, my_models, base,
                add_avg_rank=False, use_rank=False, use_std=False,
                use_baseline_pr=False, is_plot=False
            )
            dict_df[base] = df
        
        all_models = dict_df[keys[0]].index.tolist()
        dataset_results = {}
        
        # 데이터셋별로 재구성
        for dataset in data:
            ratio_labels = [key.replace('ratio_', '').replace('_AUCPR', '') for key in keys]
            model_vs_ratio_df = pd.DataFrame(index=all_models, columns=ratio_labels)
            
            for i, base_key in enumerate(keys):
                ratio_label = ratio_labels[i]
                if dataset in dict_df[base_key].columns:
                    model_vs_ratio_df[ratio_label] = dict_df[base_key][dataset]
            
            dataset_results[dataset] = model_vs_ratio_df
            
            print(f"\nDataset: {dataset.upper()}")
            print("=" * 60)
            print("Model vs Training Ratio (AUCPR)")
            print(model_vs_ratio_df.round(4))
            
            os.makedirs('metrics/train_ratio', exist_ok=True)
            model_vs_ratio_df.to_csv(f'metrics/train_ratio/{dataset}_model_vs_ratio.csv')
        
        if print_summary:
            self._print_best_models_summary(dataset_results, data)
        
        return dataset_results
    
    @staticmethod
    def _print_best_models_summary(
        dataset_results: Dict[str, pd.DataFrame], 
        data: List[str]
    ):
        """각 데이터셋/비율별 최고 성능 모델 출력"""
        print("\n" + "=" * 80)
        print("BEST MODEL PER RATIO PER DATASET")
        print("=" * 80)
        
        best_models_summary = {}
        for dataset in data:
            best_models_summary[dataset] = {}
            df = dataset_results[dataset]
            
            print(f"\n{dataset.upper()}:")
            for ratio in df.columns:
                if not df[ratio].isna().all():
                    best_model = df[ratio].idxmax()
                    best_score = df[ratio].max()
                    best_models_summary[dataset][ratio] = (best_model, best_score)
                    print(f"  Ratio {ratio}: {best_model} ({best_score:.4f})")
    
    def render_train_ratio_average(
        self, 
        pivots: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """훈련 비율별 전체 평균 성능"""
        data = ['campaign', 'nslkdd', 'census']
        models = ['MCM', 'DRL', 'Disent']
        my_models = ['MemPAE-ws-pos_query+token-d64-lr0.001-t0.1']
        keys = [
            'ratio_0.2_AUCPR', 'ratio_0.4_AUCPR', 'ratio_0.6_AUCPR',
            'ratio_0.8_AUCPR', 'ratio_1.0_AUCPR',
        ]
        
        # 각 비율별 데이터프레임 수집
        dict_df = {}
        for base in keys:
            df = self.renderer.render(
                pivots, data, models, my_models, base,
                add_avg_rank=False, use_rank=False, use_std=False,
                use_baseline_pr=False, is_plot=False
            )
            dict_df[base] = df
        
        all_models = dict_df[keys[0]].index.tolist()
        ratio_labels = [key.replace('ratio_', '').replace('_AUCPR', '') for key in keys]
        
        # 전체 평균 계산
        average_df = pd.DataFrame(index=all_models, columns=ratio_labels)
        
        for i, base_key in enumerate(keys):
            ratio_label = ratio_labels[i]
            average_df[ratio_label] = dict_df[base_key].mean(axis=1)
        
        print("=" * 60)
        print("OVERALL AVERAGE - Model vs Training Ratio (AUCPR)")
        print("=" * 60)
        print(average_df.round(4))
        
        os.makedirs('metrics/train_ratio', exist_ok=True)
        average_df.to_csv('metrics/train_ratio/overall_average_model_vs_ratio.csv')
        
        print(f"\nOverall average results saved to: metrics/train_ratio/overall_average_model_vs_ratio.csv")
        
        return average_df


# ============================================================================
# 메인 실행 로직
# ============================================================================

def main(args):
    """메인 실행 함수"""
    config = Config()
    
    # 데이터 수집 및 변환
    collector = ResultCollector(config)
    converter = DataFrameConverter(config)
    
    print("Collecting results...")
    results = collector.collect_results()
    
    print("Converting to DataFrames...")
    df_all, dfs = converter.convert_results_to_csv(results, save_csv=False)
    
    print("Creating pivot tables...")
    pivots = converter.make_pivots(dfs, save_csv=False)
    
    # 렌더링 및 분석
    renderer = ResultRenderer(config)
    specialized = SpecializedAnalysis(config, renderer)
    
    # 기본 데이터셋 및 모델 정의
    data = [
        'arrhythmia', 'breastw', 'cardio', 'cardiotocography', 'glass',
        'ionosphere', 'pima', 'wbc', 'wine', 'thyroid', 'optdigits', 
        'pendigits', 'satellite', 'campaign', 'mammography', 'satimage-2', 
        'nslkdd', 'fraud', 'shuttle', 'census',
    ]
    #  data.sort()
        
    data = [
        "wine",
        "glass",
        "wbc",
        "ionosphere",
        "arrhythmia",
        "breastw",
        "pima",
        "cardio",
        "cardiotocography",
        "thyroid",
        "optdigits",
        "satimage-2",
        "satellite",
        "pendigits",
        "mammography",
        "campaign",
        "shuttle",
        "nslkdd",
        "fraud",
        "census"
    ]
    data = [
        'arrhythmia', 'breastw', 'cardio', 'cardiotocography', 'glass',
        'ionosphere', 'pima', 'wbc', 'wine', 'thyroid', 'optdigits', 
        'pendigits', 'satellite', 'campaign', 'mammography', 'satimage-2', 
        'nslkdd', 'fraud', 'shuttle', 'census',
    ]
    data.sort()

    models = [
        'IForest', 'LOF', 'OCSVM', 'ECOD', 'KNN', 'PCA',
        'DeepSVDD', 'GOAD', 'NeuTraL', 'ICL', 'MCM', 'DRL', 'Disent', 
        'NPTAD', ####
        # 'RetAugv2', ####
        # 'RetAug',  
    ]
    
    my_models = [
        # pos sharing
        # "MemPAE-ws-d64-lr0.001-t0.1", 

        # Low rank proejction: N=F
        # "MemPAE-ws-local+global-F-sqrt_N1.0-d64-lr0.001-t0.1", 

        # Ablation: Component Analysis
        
        # "AutoEncoder", # MLP-MLP-X
        # 'MemAE-d256-lr0.001-t0.1', # MLP-MLP-O
        # 'MemAE-d256-lr0.001', # MLP-MLP-O
        # "MemAE-d64-lr0.001-t0.1", 
        # "MemAE-d64-lr0.001", 
        # "MemAE-d64-lr0.005", 
        # "MemAE-d64-lr0.005-t0.1", 
        # "MemAE-d64-lr0.01", 
        # "MemAE-d64-lr0.01-t0.1", 

        # "LATTE-50",
        # "LATTE-100",
        # "LATTE-150",
        # "LATTE-200",

        # "LATTE-power_2-50",
        # "LATTE-power_2-100",
        # "LATTE-power_2-150",
        # "LATTE-power_2-200",

        # Reviewer 4 (2nd rebuttal): Question 1 
        # "LATTE-Extended-50",
        # "LATTE-Extended-100",
        # "LATTE-Extended-150",
        # "LATTE-Extended-200",
        # "LATTE-Extended-250",
        # "LATTE-Extended-300",
        # "LATTE-Extended-350",
        # "LATTE-Extended-400",
        # "LATTE-Extended-450",
        # "LATTE-Extended-500",

        # 251130
        "LATTE-patience-tuned", 
        # "MemPAE-Full_rank",
        # 'LATTE-Full_rank-no_dec-d64-lr0.001-g0.99-t0.1-p20',
        # 'MemSet',
        # 'MemSet-use_pos',


        # "MemPAE-ws-local+global-F-sqrt_N1.0-d64-lr0.001-t0.1", 
        # "LATTE-no_mem-full_rank",
        # "LATTE-no_mem-full_rank-v2",
        # "LATTE-no_mem-full_rank-gamma1.0",
        # "LATTE-no_mem-full_rank-p200-gamma1.0",
        # 'LATTE-Full_rank-no_dec-d64-lr0.001-g1.0-t0.1-p20',
        # 'LATTE-Full_rank-no_dec-d64-lr0.001-g1.0-t1.0-p20',
        
        # 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1', # Attn-Attn-O
        
        # pendigits  mammography  campaign  shuttle  nslkdd   fraud  census
        # "LATTE-patience20-delta0.000001", # 
        # pima, cardio, cardiotography, thyroid, opdigits, satimage-2, satellite,
        # "LATTE-patience10-delta0.0001", 
        # wine, glass, wbc, ionosphere, arrhythmia, breastw
        # "LATTE-patience10-delta0.005",         


        # "MemPAE-mlp-mlp-v3-no_mem", # MLP-MLP-X
        # "MemPAE-mlp-mlp-v3", # MLP-MLP-O
        # "MemPAE-attn-mlp-no_mem", # Attn-MLP-X
        # "MemPAE-attn-mlp", # Attn-MLP-O
        # "MemPAE-mlp-attn-no_mem",
        # 'MemPAE-mlp-attn',
        # "MemPAE-attn-attn-no_mem", # Attn-Attn-X
        # 'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1', # Attn-Attn-O
        # "MemPAE-attn-mlp-no_mem-v2", # Attn-MLP-X
        # "MemPAE-attn-mlp-no_mem-lr0.01",
        # "MemPAE-attn-mlp-no_mem-ws-lr0.01",
        # "MemPAE-attn-mlp-ws-lr0.01",


        # "PAE-ws-pos_query+token-d64-lr0.001", # Attn-Attn-X
        # "MemPAE-attn-attn-no_mem-v2", # Attn-Attn-X


        
        # "MemPAE-ws-local+global-sqrt_F1.0-sqrt_N1.0-mlp_enc_mixer-d64-lr0.001-t0.1",
        # "MemPAE-ws-local+global-sqrt_F1.0-sqrt_N1.0-mlp_dec_mixer-d64-lr0.001-t0.1",
    ]
    
    keys = [
        'ratio_1.0_AUCROC', 
        'ratio_1.0_AUCPR', 
        'ratio_1.0_f1'
    ]
    
    for base in keys:
        print(f"\nRendering {base}...")
        renderer.render(
            pivots, data, models, my_models, base,
            add_avg_rank=True, use_rank=False, use_std=args.use_std,
            use_baseline_pr=True, is_plot=False,
            use_temp=args.use_temp, 
            use_top_k=args.use_top_k, 
            use_hpo_memory_latent=args.use_hpo_memory_latent,
            use_hpo_memory_latent_top_k=args.use_hpo_memory_latent_top_k,
        )

    if args.stat_test:

        test_models = [
            'IForest', 'LOF', 'OCSVM', 'ECOD', 'KNN', 'PCA',
            'DeepSVDD', 'GOAD', 'NeuTraL', 'ICL', 'MCM', 'DRL', 'Disent', 
        ]

        my_test_models = [
            'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1', # Attn-Attn-O
        ]

        from statistical_test import run_statistical_analysis
        tr, metr = base.split('_')[1], base.split('_')[2]
        k_mean = f"ratio_{tr}_{metr}_mean"
        df_for_test = pivots[k_mean][data].copy()
        highlight_models = [m for m in my_models if m in df_for_test.index]
        # 모델 순서 재배열
        first = [m for m in test_models if m in df_for_test.index]
        rest = [m for m in my_test_models if m in df_for_test.index]
        order = first + rest
        df_for_test = df_for_test.loc[order]
        from cd_diagram import draw_cd_diagram_from_pivot
        print(df_for_test.index)
        print(df_for_test.columns)
        draw_cd_diagram_from_pivot(
            df_pivot=df_for_test,
            alpha=0.05,
        )

        # results_stat = run_statistical_analysis(
        #     df_mean=df_for_test,
        #     alpha=0.05,
        #     plot=True,
        #     save_dir='metrics/statistical_tests',
        #     filename_prefix=f'{base}_critical_difference',
        #     highlight_models=highlight_models
        # )
        
        # # 결과를 CSV로 저장
        # import os
        # os.makedirs('metrics/statistical_tests', exist_ok=True)
        
        # # 평균 랭킹 저장
        # results_stat['avg_ranks'].to_csv(
        #     f'metrics/statistical_tests/{base}_avg_ranks.csv',
        #     header=['Average Rank']
        # )


        
    # 특수 분석
    if args.train_ratio:
        print("\n" + "=" * 80)
        print("Training Ratio Analysis")
        print("=" * 80)
        specialized.render_train_ratio(pivots, False)
        specialized.render_train_ratio_average(pivots)
    
    # Synthetic 데이터 분석
    if args.synthetic:
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
            # "MemPAE-ws-local+global-sqrt_F1.0-sqrt_N1.0-mlp_enc_dec_mixer-d64-lr0.001-t0.1",
            # "MemPAE-mlp-mlp-d256-lr0.1",
            # "MemPAE-mlp-mlp-d256-lr0.01",
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
    # Contamination 분석
    if args.contamination:
        print("\n" + "=" * 80)
        print("Contamination Analysis")
        print("=" * 80)
        
        models_contam = ['KNN', 'MCM', 'DRL', 'Disent']
        my_models_contam = [
            'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1',
            "MemPAE-ws-local+global-sqrt_F1.0-sqrt_N1.0-mlp_enc_mixer-d64-lr0.001-t0.1",
            "MemPAE-ws-local+global-sqrt_F1.0-sqrt_N1.0-mlp_dec_mixer-d64-lr0.001-t0.1",
        ]
        
        dataname_list_contam = [
            'pima', 'arrhythmia', 'glass', 'wbc', 'wine',
            'campaign', 'ionosphere', 'shuttle', 'satellite',
        ]
        
        contamination_ratio = ['contam0.01', 'contam0.03', 'contam0.05']
        keys = ['ratio_1.0_AUCPR']
        
        for dataname in dataname_list_contam:
            synthetic_data_contam = [dataname.split('_')[-1]]
            print(f"\n{'='*80}")
            print(f"Dataset: {dataname}")
            print(f"{'='*80}")
            
            for contamination in contamination_ratio:
                file_name = f"{dataname}_{contamination}"
                synthetic_data_contam.append(file_name)
            
            for base in keys:
                df_render = renderer.render(
                    pivots, synthetic_data_contam, models_contam, my_models_contam, base,
                    add_avg_rank=True, use_rank=False, use_std=True,
                    use_baseline_pr=False, use_alias=True, is_temp_tune=False,
                    is_synthetic=True, synthetic_type=f"{dataname}_contamination",
                    is_plot=False, is_print=True,
                )
    
    # Hyperparameter Ratio 분석
    if args.hp_ratio:
        print("\n" + "=" * 80)
        print("Hyperparameter Ratio Analysis")
        print("=" * 80)
        
        data_hp = [
            'arrhythmia', 'breastw', 'cardio', 'cardiotocography', 'glass',
            'ionosphere', 'pima', 'wbc', 'wine', 'thyroid',
            'optdigits', 'pendigits', 'satellite', 'campaign', 'mammography',
            'satimage-2', 'nslkdd', 'fraud', 'shuttle', 'census',
        ]
        data_hp.sort()
        
        models_hp = ['KNN']
        my_models_hp = [
            'MemPAE-ws-pos_query+token-latent_ratio0.5-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-latent_ratio2.0-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-latent_ratio4.0-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-latent_ratio8.0-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-memory_ratio0.5-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-memory_ratio2.0-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-memory_ratio4.0-d64-lr0.001-t0.1',
            'MemPAE-ws-pos_query+token-memory_ratio8.0-d64-lr0.001-t0.1',
        ]
        
        base_hp = 'ratio_1.0_AUCPR'
        renderer.render(
            pivots, data_hp, models_hp, my_models_hp, base_hp,
            add_avg_rank=True, use_rank=False, use_std=False,
            use_baseline_pr=True, is_temp_tune=False, is_sort=False,
            is_plot=True, is_print=True,
            
        )
  
    print("\nAnalysis complete!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='실험 결과 분석 스크립트')
    parser.add_argument('--use_std', action='store_true', help='show std.')
    parser.add_argument('--synthetic', action='store_true', help='합성 데이터 분석 수행')
    parser.add_argument('--contamination', action='store_true', help='오염 비율 분석 수행')
    parser.add_argument('--hp_ratio', action='store_true', help='하이퍼파라미터 비율 분석 수행')
    parser.add_argument('--npt', action='store_true', help='NPT-AD와 비교')
    parser.add_argument('--train_ratio', action='store_true', help='훈련 비율별 분석 수행')
    parser.add_argument('--synthetic_type', type=str, default='dependency', help='합성 데이터 타입')
    parser.add_argument('--use_temp', action='store_true', help='temperature')
    parser.add_argument('--use_top_k', action='store_true', help='topk')
    parser.add_argument('--use_hpo_memory_latent', action='store_true', help='memory latent')
    parser.add_argument('--use_hpo_memory_latent_top_k', action='store_true', help='memory latent topk')
    parser.add_argument('--stat_test', action='store_true', help='Criticla Difference Diagram.')
    
    args = parser.parse_args()
    main(args)