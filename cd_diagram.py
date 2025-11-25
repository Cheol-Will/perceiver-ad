# Based on: Hassan Ismail Fawaz et al.
# Adapted for model comparison across multiple datasets

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = 'Arial'

import operator
import math
from scipy.stats import wilcoxon, friedmanchisquare
import networkx
from typing import Optional, List, Tuple, Dict


def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, labels=False, **kwargs):
    """
    Draws a CD graph, which is used to display the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        p_values: list of tuples (classifier_1, classifier_2, p_value, significant)
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional): if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension).
        labels (bool, optional): if set to `True`, the calculated avg rank
            values will be displayed
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """Returns only nth element in a list."""
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """List location in list of list structure."""
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    sums = avranks
    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4
    k = len(sums)
    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25
    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """Input is a list of pairs of points."""
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=2)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom", size=16)

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=linewidth)
        if labels:
            text(textspace + 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="right", va="center", size=10)
        text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=16)

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=linewidth)
        if labels:
            text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.4f'), ha="left", va="center", size=10)
        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
             ha="left", va="center", size=16)

    # draw no significant lines
    start = cline + 0.2
    side = -0.02
    height = 0.1

    # get the cliques
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    
    for clq in cliques:
        if len(clq) == 1:
            continue
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        line([(rankpos(ssums[min_idx]) - side, start),
              (rankpos(ssums[max_idx]) + side, start)],
             linewidth=linewidth_sign)
        start += height

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"CD diagram saved to: {filename}")
    
    plt.close()


def form_cliques(p_values, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:  # if not significant
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)


def wilcoxon_holm(alpha=0.05, df_perf=None):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis
    
    Args:
        alpha: significance level
        df_perf: DataFrame with columns ['classifier_name', 'dataset_name', 'accuracy']
    
    Returns:
        p_values: list of tuples (classifier_1, classifier_2, p_value, significant)
        average_ranks: Series with average ranks
        max_nb_datasets: number of datasets
    """
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['classifier_name']).size()}).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts['count'].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['classifier_name'])
    
    print(f"\nClassifiers: {classifiers}")
    print(f"Number of datasets: {max_nb_datasets}")
    
    # test the null hypothesis using friedman before doing a post-hoc analysis
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
        for c in classifiers))[1]
    
    print(f"Friedman test p-value: {friedman_p_value:.6f}")
    
    if friedman_p_value >= alpha:
        print('The null hypothesis over the entire classifiers cannot be rejected')
        return None, None, None
    
    # get the number of classifiers
    m = len(classifiers)
    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    
    # loop through the algorithms to compare pairwise
    for i in range(m - 1):
        classifier_1 = classifiers[i]
        perf_1 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_1]['accuracy'],
                          dtype=np.float64)
        for j in range(i + 1, m):
            classifier_2 = classifiers[j]
            perf_2 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_2]['accuracy'],
                              dtype=np.float64)
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
            p_values.append((classifier_1, classifier_2, p_value, False))
    
    # get the number of hypothesis
    k = len(p_values)
    # sort the list in ascending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            break
    
    # compute the average ranks
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)].sort_values(
        ['classifier_name', 'dataset_name'])
    
    rank_data = np.array(sorted_df_perf['accuracy']).reshape(m, max_nb_datasets)
    
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers),
                            columns=np.unique(sorted_df_perf['dataset_name']))

    # number of wins
    dfff = df_ranks.rank(ascending=False)
    print("\nNumber of wins (rank=1) per classifier:")
    print(dfff[dfff == 1.0].sum(axis=1))

    # average the ranks
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    
    return p_values, average_ranks, max_nb_datasets


def draw_cd_diagram(df_perf=None, alpha=0.05, title=None, labels=False, filename='cd-diagram.png'):
    """
    Draws the critical difference diagram given the DataFrame of performances
    
    Args:
        df_perf: DataFrame with columns ['classifier_name', 'dataset_name', 'accuracy']
        alpha: significance level
        title: plot title
        labels: whether to show rank values
        filename: output filename
    """
    p_values, average_ranks, max_nb_datasets = wilcoxon_holm(df_perf=df_perf, alpha=alpha)
    
    if p_values is None:
        print("Cannot draw CD diagram - Friedman test not significant")
        return None
    
    print("\nAverage ranks:")
    print(average_ranks)
    
    print("\nPairwise comparisons:")
    for p in p_values:
        sig_str = "significant" if p[3] else "not significant"
        print(f"{p[0]} vs {p[1]}: p={p[2]:.6f} ({sig_str})")

    graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
                cd=None, reverse=True, width=9, textspace=1.5, labels=labels,
                filename=filename)

    font = {'family': 'sans-serif',
            'color': 'black',
            'weight': 'normal',
            'size': 22}
    
    if title:
        plt.title(title, fontdict=font, y=0.9, x=0.5)
    
    return average_ranks, p_values


def convert_pivot_to_long_format(df_pivot: pd.DataFrame, metric_name: str = 'auc-pr') -> pd.DataFrame:
    """
    피벗 테이블 (모델 x 데이터셋)을 long format으로 변환
    
    Args:
        df_pivot: 모델(행) x 데이터셋(컬럼) DataFrame
        metric_name: 메트릭 이름 (기본값: 'accuracy')
    
    Returns:
        DataFrame with columns ['classifier_name', 'dataset_name', metric_name]
    """
    # AVG_RANK, AVG_AUC 등 집계 컬럼 제외
    exclude_cols = ['AVG_RANK', 'AVG_AUC', 'AVG_TIER']
    data_cols = [col for col in df_pivot.columns if col not in exclude_cols]

    df_subset = df_pivot[data_cols].copy()
    df_subset = df_subset.rename(index={'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1': 'LATTE'}) 
    
    # long format으로 변환
    df_long = df_subset.reset_index().melt(
        id_vars='model',
        var_name='dataset',
        value_name=metric_name
    )
    df_long = df_long.rename(columns={'model': 'classifier_name'})
    df_long = df_long.rename(columns={'dataset': 'dataset_name'})
    
    # NaN 제거
    df_long = df_long.dropna()
    
    return df_long


def draw_cd_diagram_from_pivot(
    df_pivot: pd.DataFrame,
    alpha: float = 0.05,
    title: Optional[str] = None,
    labels: bool = True,
    filename: str = 'cd-diagram.png'
):
    """
    피벗 테이블로부터 직접 CD diagram 생성
    
    Args:
        df_pivot: 모델(행) x 데이터셋(컬럼) 성능 DataFrame
        alpha: 유의수준
        title: 그래프 제목
        labels: 랭킹 값 표시 여부
        filename: 저장 파일명
    
    Returns:
        average_ranks, p_values
    """
    # long format으로 변환
    df_long = convert_pivot_to_long_format(df_pivot, metric_name='accuracy')
    
    print(f"\nConverted to long format: {len(df_long)} rows")
    print(f"Models: {df_long['classifier_name'].nunique()}")
    print(f"Datasets: {df_long['dataset_name'].nunique()}")
    
    # CD diagram 그리기
    return draw_cd_diagram(df_perf=df_long, alpha=alpha, title=title, 
                          labels=labels, filename=filename)


if __name__ == "__main__":
    # 테스트 예제
    np.random.seed(42)
    
    # 샘플 데이터 생성
    models = ['Model_A', 'Model_B', 'Model_C', 'Model_D', 'Model_E']
    datasets = [f'Dataset_{i}' for i in range(10)]
    
    # 성능 차이가 있는 데이터
    base_scores = {'Model_A': 0.90, 'Model_B': 0.85, 'Model_C': 0.82, 
                   'Model_D': 0.78, 'Model_E': 0.75}
    
    data = {}
    for ds in datasets:
        data[ds] = []
        for model in models:
            score = np.random.normal(base_scores[model], 0.03)
            score = np.clip(score, 0, 1)
            data[ds].append(score)
    
    df_pivot = pd.DataFrame(data, index=models).T.T
    
    print("Sample data (pivot format):")
    print(df_pivot.head())
    
    # CD diagram 생성
    draw_cd_diagram_from_pivot(
        df_pivot,
        alpha=0.05,
        title='Model Comparison',
        labels=True,
        filename='test_cd_diagram.png'
    )