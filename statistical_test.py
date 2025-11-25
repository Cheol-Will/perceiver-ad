import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import friedmanchisquare
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class StatisticalTester:
    """통계 검정 클래스 (Friedman Test + Nemenyi Post-hoc Test)"""
    
    def __init__(self, df_mean: pd.DataFrame, alpha: float = 0.05):
        """
        Parameters:
        -----------
        df_mean : pd.DataFrame
            모델(행) × 데이터셋(컬럼) 형태의 성능 점수 DataFrame
        alpha : float
            유의수준 (기본값: 0.05)
        """
        self.df_mean = df_mean.copy()
        self.alpha = alpha
        self.n_models = len(df_mean)
        self.n_datasets = len(df_mean.columns)
        
        # 'AVG_RANK', 'AVG_AUC' 등의 집계 컬럼 제외
        exclude_cols = ['AVG_RANK', 'AVG_AUC', 'AVG_TIER']
        self.dataset_cols = [col for col in df_mean.columns if col not in exclude_cols]
        
        # 데이터셋 컬럼만 추출
        self.df_datasets = df_mean[self.dataset_cols].copy()
        self.n_datasets = len(self.dataset_cols)
        
    def compute_ranks(self) -> pd.DataFrame:
        """각 데이터셋에서 모델의 랭킹 계산 (높은 점수 = 낮은 순위)"""
        # 각 데이터셋(컬럼)에서 내림차순 랭킹 (1위가 가장 좋은 모델)
        ranks = self.df_datasets.rank(axis=0, ascending=False, method='average')
        return ranks
    
    def friedman_test(self) -> Tuple[float, float, bool]:
        """
        Friedman Test 수행
        
        Returns:
        --------
        statistic : float
            Friedman 검정 통계량
        p_value : float
            p-value
        significant : bool
            귀무가설 기각 여부
        """
        # 각 데이터셋을 하나의 블록으로 간주
        # 데이터를 리스트 형태로 변환 (각 모델의 성능을 데이터셋별로)
        data_for_friedman = []
        for model in self.df_datasets.index:
            model_scores = self.df_datasets.loc[model].values
            # NaN 제거
            model_scores = model_scores[~np.isnan(model_scores)]
            data_for_friedman.append(model_scores)
        
        # Friedman test 수행
        statistic, p_value = friedmanchisquare(*data_for_friedman)
        significant = p_value < self.alpha
        
        return statistic, p_value, significant
    
    def nemenyi_test(self, ranks: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Nemenyi Post-hoc Test 수행
        
        Parameters:
        -----------
        ranks : pd.DataFrame
            각 데이터셋에서의 모델 랭킹
            
        Returns:
        --------
        pairwise_diff : pd.DataFrame
            모델 간 평균 랭킹 차이 행렬
        cd : float
            Critical Difference 값
        """
        # 각 모델의 평균 랭킹 계산
        avg_ranks = ranks.mean(axis=1)
        
        # Critical Difference 계산
        # CD = q_alpha * sqrt(k(k+1) / (6N))
        # 여기서 k = 모델 수, N = 데이터셋 수
        k = self.n_models
        N = self.n_datasets
        
        # Nemenyi test에서 사용하는 q_alpha 값 (Studentized range statistic)
        # 근사값 사용 (정확한 값은 테이블 참조 필요)
        q_alpha = self._get_q_alpha(k, self.alpha)
        
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
        
        # 모델 간 평균 랭킹 차이 계산
        pairwise_diff = pd.DataFrame(
            np.abs(avg_ranks.values[:, None] - avg_ranks.values[None, :]),
            index=avg_ranks.index,
            columns=avg_ranks.index
        )
        
        return pairwise_diff, cd
    
    def _get_q_alpha(self, k: int, alpha: float) -> float:
        """
        Studentized range statistic 근사값
        
        Parameters:
        -----------
        k : int
            모델(그룹) 수
        alpha : float
            유의수준
            
        Returns:
        --------
        q_alpha : float
            Critical value
        """
        # Nemenyi test의 critical value 근사
        # 정확한 값은 statistical tables를 참조해야 하지만, 
        # 여기서는 일반적으로 사용되는 근사값 제공
        
        # alpha=0.05일 때의 근사값 (k에 따라)
        if alpha == 0.05:
            q_values = {
                2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
                7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
                12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426,
                17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544
            }
        elif alpha == 0.10:
            q_values = {
                2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589,
                7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920, 11: 2.978,
                12: 3.030, 13: 3.077, 14: 3.120, 15: 3.160, 16: 3.197,
                17: 3.231, 18: 3.263, 19: 3.293, 20: 3.321
            }
        else:
            # 기본값 (alpha=0.05 근사)
            q_values = {k: 2.5 + 0.5 * np.log(k) for k in range(2, 21)}
        
        if k in q_values:
            return q_values[k]
        else:
            # k > 20인 경우 보간
            return 2.5 + 0.5 * np.log(k)
    
    def get_significant_pairs(
        self, 
        pairwise_diff: pd.DataFrame, 
        cd: float
    ) -> List[Tuple[str, str]]:
        """
        통계적으로 유의한 차이가 없는 모델 쌍 찾기
        
        Parameters:
        -----------
        pairwise_diff : pd.DataFrame
            모델 간 평균 랭킹 차이
        cd : float
            Critical Difference
            
        Returns:
        --------
        non_significant_pairs : List[Tuple[str, str]]
            유의한 차이가 없는 모델 쌍 리스트
        """
        non_significant_pairs = []
        
        for i in range(len(pairwise_diff)):
            for j in range(i + 1, len(pairwise_diff)):
                model_i = pairwise_diff.index[i]
                model_j = pairwise_diff.index[j]
                diff = pairwise_diff.iloc[i, j]
                
                # 차이가 CD보다 작으면 유의한 차이 없음
                if diff < cd:
                    non_significant_pairs.append((model_i, model_j))
        
        return non_significant_pairs
    
    def run_full_test(self, verbose: bool = True) -> Dict:
        """
        전체 통계 검정 수행 (Friedman + Nemenyi)
        
        Parameters:
        -----------
        verbose : bool
            결과 출력 여부
            
        Returns:
        --------
        results : Dict
            검정 결과 딕셔너리
        """
        results = {}
        
        # 1. 랭킹 계산
        ranks = self.compute_ranks()
        avg_ranks = ranks.mean(axis=1).sort_values()
        results['ranks'] = ranks
        results['avg_ranks'] = avg_ranks
        
        # 2. Friedman Test
        friedman_stat, friedman_p, significant = self.friedman_test()
        results['friedman_statistic'] = friedman_stat
        results['friedman_pvalue'] = friedman_p
        results['friedman_significant'] = significant
        
        if verbose:
            print("=" * 80)
            print("FRIEDMAN TEST")
            print("=" * 80)
            print(f"Statistic: {friedman_stat:.4f}")
            print(f"P-value: {friedman_p:.6f}")
            print(f"Significant (α={self.alpha}): {significant}")
            print()
        
        # 3. Nemenyi Post-hoc Test (Friedman이 유의한 경우)
        if significant:
            pairwise_diff, cd = self.nemenyi_test(ranks)
            non_sig_pairs = self.get_significant_pairs(pairwise_diff, cd)
            
            results['pairwise_diff'] = pairwise_diff
            results['critical_difference'] = cd
            results['non_significant_pairs'] = non_sig_pairs
            
            if verbose:
                print("=" * 80)
                print("NEMENYI POST-HOC TEST")
                print("=" * 80)
                print(f"Critical Difference (CD): {cd:.4f}")
                print(f"\nAverage Ranks:")
                for model, rank in avg_ranks.items():
                    print(f"  {model:50s}: {rank:.3f}")
                print(f"\nNon-significant pairs (diff < CD={cd:.4f}):")
                for pair in non_sig_pairs:
                    diff = pairwise_diff.loc[pair[0], pair[1]]
                    print(f"  {pair[0]:30s} <-> {pair[1]:30s} (diff: {diff:.3f})")
                print()
        else:
            if verbose:
                print("Friedman test not significant. Skipping post-hoc test.")
                print()
        
        return results


class CriticalDifferencePlotter:
    """Critical Difference Diagram 시각화 클래스"""
    
    def __init__(self, avg_ranks: pd.Series, cd: float, non_sig_pairs: List[Tuple]):
        """
        Parameters:
        -----------
        avg_ranks : pd.Series
            각 모델의 평균 랭킹 (오름차순 정렬)
        cd : float
            Critical Difference 값
        non_sig_pairs : List[Tuple]
            통계적으로 유의하지 않은 모델 쌍
        """
        self.avg_ranks = avg_ranks.sort_values()
        self.cd = cd
        self.non_sig_pairs = non_sig_pairs
        self.models = self.avg_ranks.index.tolist()
        self.n_models = len(self.models)
        
    def plot(
        self, 
        figsize: Tuple[int, int] = (12, 6),
        title: str = "Critical Difference Diagram (Nemenyi Test)",
        filename: Optional[str] = None,
        highlight_models: Optional[List[str]] = None
    ):
        """
        CD Diagram 그리기
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure 크기
        title : str
            그래프 제목
        filename : str, optional
            저장할 파일명
        highlight_models : List[str], optional
            강조할 모델 리스트
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 축 설정
        max_rank = self.avg_ranks.max()
        min_rank = self.avg_ranks.min()
        rank_range = max_rank - min_rank
        
        # x축: 평균 랭킹
        ax.set_xlim(min_rank - rank_range * 0.1, max_rank + rank_range * 0.1)
        # y축: 모델별 위치
        ax.set_ylim(-1, self.n_models)
        
        # 모델 이름 단순화
        simplified_names = []
        for model in self.models:
            if 'MemPAE' in model:
                simplified_names.append('MemPAE')
            elif 'PAE' in model:
                simplified_names.append('PAE')
            elif 'PDRL' in model:
                simplified_names.append('PDRL')
            else:
                simplified_names.append(model)
        
        # 모델별 y 좌표
        y_positions = np.arange(self.n_models)
        
        # 1. 모델명과 랭킹 표시
        for i, (model, simplified) in enumerate(zip(self.models, simplified_names)):
            rank = self.avg_ranks[model]
            color = 'red' if highlight_models and model in highlight_models else 'black'
            fontweight = 'bold' if highlight_models and model in highlight_models else 'normal'
            
            # 모델명 (왼쪽)
            ax.text(min_rank - rank_range * 0.05, y_positions[i], simplified,
                   ha='right', va='center', fontsize=12, color=color, fontweight=fontweight)
            
            # 랭킹 점
            ax.plot(rank, y_positions[i], 'o', markersize=10, color=color, zorder=3)
            
            # 랭킹 값 (오른쪽)
            ax.text(max_rank + rank_range * 0.05, y_positions[i], f'{rank:.2f}',
                   ha='left', va='center', fontsize=10, color=color)
        
        # 2. 유의하지 않은 쌍 연결 (수평선)
        # 모델 인덱스 매핑
        model_to_idx = {model: i for i, model in enumerate(self.models)}
        
        # 연결선 그룹화 (clique detection)
        cliques = self._find_cliques(self.non_sig_pairs)
        
        # 각 clique마다 수평선 그리기
        line_height_offset = 0.15
        for clique_idx, clique in enumerate(cliques):
            if len(clique) < 2:
                continue
            
            # clique 내 모델들의 인덱스와 랭킹
            indices = [model_to_idx[m] for m in clique]
            ranks = [self.avg_ranks[m] for m in clique]
            
            # 최소/최대 랭킹
            min_rank_in_clique = min(ranks)
            max_rank_in_clique = max(ranks)
            
            # 수평선 높이 (모델들의 평균 y 위치 기준)
            avg_y = np.mean([y_positions[i] for i in indices])
            
            # 수평선 그리기
            line_y = avg_y - line_height_offset * (clique_idx % 3)
            ax.plot([min_rank_in_clique, max_rank_in_clique], [line_y, line_y],
                   'b-', linewidth=3, alpha=0.6, zorder=1)
            
            # 각 모델에서 수평선으로 수직선 연결
            for idx in indices:
                ax.plot([self.avg_ranks[self.models[idx]], self.avg_ranks[self.models[idx]]],
                       [y_positions[idx], line_y], 'b-', linewidth=1, alpha=0.4, zorder=1)
        
        # 3. Critical Difference 표시
        cd_y = -0.5
        cd_x_center = (min_rank + max_rank) / 2
        ax.plot([cd_x_center - self.cd/2, cd_x_center + self.cd/2], 
               [cd_y, cd_y], 'r-', linewidth=3, label=f'CD = {self.cd:.3f}')
        ax.plot([cd_x_center - self.cd/2, cd_x_center - self.cd/2], 
               [cd_y - 0.1, cd_y + 0.1], 'r-', linewidth=2)
        ax.plot([cd_x_center + self.cd/2, cd_x_center + self.cd/2], 
               [cd_y - 0.1, cd_y + 0.1], 'r-', linewidth=2)
        ax.text(cd_x_center, cd_y - 0.3, f'Critical Difference = {self.cd:.3f}',
               ha='center', fontsize=11, color='red', fontweight='bold')
        
        # 축 설정
        ax.set_xlabel('Average Rank', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 범례
        blue_patch = mpatches.Patch(color='blue', alpha=0.6, label='No significant difference')
        red_line = mpatches.Patch(color='red', label=f'CD = {self.cd:.3f}')
        ax.legend(handles=[blue_patch, red_line], loc='upper right', fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"CD diagram saved to: {filename}")
        
        plt.show()
        plt.close()
    
    def _find_cliques(self, pairs: List[Tuple[str, str]]) -> List[List[str]]:
        """
        연결된 모델 그룹(clique) 찾기
        
        Parameters:
        -----------
        pairs : List[Tuple[str, str]]
            연결된 모델 쌍
            
        Returns:
        --------
        cliques : List[List[str]]
            연결된 모델 그룹들
        """
        if not pairs:
            return []
        
        # 그래프 구성
        from collections import defaultdict
        graph = defaultdict(set)
        for m1, m2 in pairs:
            graph[m1].add(m2)
            graph[m2].add(m1)
        
        # DFS로 연결 요소 찾기
        visited = set()
        cliques = []
        
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node in graph:
            if node not in visited:
                component = []
                dfs(node, component)
                if len(component) > 1:
                    cliques.append(sorted(component))
        
        return cliques


def run_statistical_analysis(
    df_mean: pd.DataFrame,
    alpha: float = 0.05,
    plot: bool = True,
    save_dir: str = 'metrics',
    filename_prefix: str = 'cd_diagram',
    highlight_models: Optional[List[str]] = None
) -> Dict:
    """
    통계 검정 및 CD diagram 생성 전체 파이프라인
    
    Parameters:
    -----------
    df_mean : pd.DataFrame
        모델(행) × 데이터셋(컬럼) 성능 데이터
    alpha : float
        유의수준
    plot : bool
        CD diagram 그릴지 여부
    save_dir : str
        저장 디렉토리
    filename_prefix : str
        파일명 접두사
    highlight_models : List[str], optional
        강조할 모델 리스트
        
    Returns:
    --------
    results : Dict
        검정 결과
    """
    import os
    
    # 1. 통계 검정
    tester = StatisticalTester(df_mean, alpha=alpha)
    results = tester.run_full_test(verbose=True)
    
    # 2. CD diagram 그리기
    if plot and results.get('friedman_significant', False):
        plotter = CriticalDifferencePlotter(
            avg_ranks=results['avg_ranks'],
            cd=results['critical_difference'],
            non_sig_pairs=results['non_significant_pairs']
        )
        
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'{filename_prefix}.png')
        
        plotter.plot(
            figsize=(14, max(6, len(results['avg_ranks']) * 0.5)),
            filename=filename,
            highlight_models=highlight_models
        )
    elif not results.get('friedman_significant', False):
        print("Friedman test not significant. CD diagram not generated.")
    
    return results


if __name__ == "__main__":
    # 테스트 예제
    np.random.seed(42)
    
    # 샘플 데이터 생성 (모델 × 데이터셋)
    models = ['Model_A', 'Model_B', 'Model_C', 'Model_D', 'Model_E']
    datasets = [f'Dataset_{i}' for i in range(10)]
    
    # 무작위 성능 점수
    data = np.random.rand(len(models), len(datasets)) * 0.3 + 0.7
    df_test = pd.DataFrame(data, index=models, columns=datasets)
    
    print("Test Data:")
    print(df_test)
    print()
    
    # 통계 분석 실행
    results = run_statistical_analysis(
        df_test,
        alpha=0.05,
        plot=True,
        filename_prefix='test_cd_diagram',
        highlight_models=['Model_A', 'Model_E']
    )