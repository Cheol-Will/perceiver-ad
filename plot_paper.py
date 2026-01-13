import os, json, glob
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 
import seaborn as sns
from summary import collect_results, convert_results_to_csv, make_pivots, render
    
pd.set_option('display.max_rows', None)

use_latex = False
if use_latex:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
else:
    plt.rcParams.update({
        'font.family': 'serif',
    })

sns.set_theme(style="ticks", context="paper")

def plot_hp_sen():
    aucpr_vs_latent = [
        [0.5, 0.7019],
        [1.0, 0.7124],
        [2.0, 0.7040],
        [4.0, 0.7085],
        [8.0, 0.7010],
    ]
    aucpr_vs_memory = [
        [0.5, 0.7044],
        [1.0, 0.7124],
        [2.0, 0.7096],
        [4.0, 0.7071],
        [8.0, 0.7061],
    ]
    aucpr_vs_temperature = [
        [1, 0.7004],
        [0.5, 0.7042],
        [0.1, 0.7124],
        [0.05, 0.7048],
        [0.01, 0.6842],
    ]
    aucpr_vs_depth = [
        [0, 0.7004],
        [2, 0.7042],
        [4, 0.7124],
        [6, 0.7070],
    ]

    aucpr_vs_top_k_mbt = [
        [5, 0.7245],
        [10, 0.7253],
        [16, 0.7258],
        [32, 0.7246],
        ['soft-knn', 0.7079],
    ]

    aucpr_vs_hidden_dim_mbt = [
        [32, 0.7076],
        [64, 0.7180],
        [128, 0.7245],
    ]

    aucpr_vs_mixup_moco = [
        [0.1, 0.6874],
        [0.2, 0.6843],
        [0.3, 0.6762],
    ]

    aucpr_vs_cont_moco = [
        [0.1, 0.6874],
        [1.0, 0.6765],
    ]



    def _plot_hp_vs_aucpr(metrics, xlabel, color, y_label = ' '):
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        ax1 = ax
        x1, y1 = zip(*metrics)
        x1_positions = range(len(x1)) 
        ax1.plot(x1_positions, y1, marker='o', linestyle='-', color=color, label='Average AUC-PR')
        ax1.set_xticks(x1_positions) 
        ax1.set_xticklabels(x1, fontsize=12)      
        ax1.tick_params(axis='x', labelsize=10)
        ax1.tick_params(axis='y', labelsize=10)
        # ax1.grid(True, axis='y', linestyle='--', linewidth=0.5) # 
        ax1.set_ylabel(y_label, fontsize=16, labelpad=10)
        ax1.set_xlabel(f'{xlabel}', fontsize=16, labelpad=10)

        plt.tight_layout()
        sns.despine(fig)

        save_dir = 'results_analysis_paper/ablation_study/'
        os.makedirs(save_dir, exist_ok=True)

        xlabel = xlabel.replace(' ', '_')        
        png_path = os.path.join(save_dir, f'hp_sensitivity_{xlabel}.png')
        # pdf_path = os.path.join(save_dir, f'hp_sensitivity_{xlabel}.pdf')

        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        # plt.savefig(pdf_path, bbox_inches='tight')
        plt.show()

        print(f"Plot saved into {png_path}")

    _plot_hp_vs_aucpr(aucpr_vs_latent, '# Latents (scale)', color='C0', y_label='AUC-PR')
    _plot_hp_vs_aucpr(aucpr_vs_memory, 'Memory size (scale)', color='C1')
    _plot_hp_vs_aucpr(aucpr_vs_temperature, 'Temperature', color='C2')
    _plot_hp_vs_aucpr(aucpr_vs_depth, 'Depth', color='C3')
    _plot_hp_vs_aucpr(aucpr_vs_top_k_mbt, 'Top-k', color='C4')
    _plot_hp_vs_aucpr(aucpr_vs_hidden_dim_mbt, 'Hidden Dim', color='C4')
    _plot_hp_vs_aucpr(aucpr_vs_mixup_moco, 'mixup_rate', color='C4')
    _plot_hp_vs_aucpr(aucpr_vs_cont_moco, 'contrastive weight', color='C4')


    def plot_hp_sen_1x4():
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        def _plot(ax, metrics, xlabel, ylabel, color):
            x, y = zip(*metrics)
            x_positions = range(len(x))
            ax.plot(
                x_positions, 
                y, 
                marker='o', 
                linestyle='-', 
                color=color, 
                label='Average AUC-PR',
                linewidth=2.5,
                markersize=8,
                markeredgewidth=1.5,    
            )
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x, fontsize=14)
            ax.tick_params(axis='x', length=0, labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_xlabel(xlabel, fontsize=16)
            ax.set_xlabel(xlabel, fontsize=16, labelpad=10)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=16, labelpad=10)

        _plot(axes[0], aucpr_vs_latent, 'Latent Number Scale Factor', 'AUC-PR', 'C0')
        _plot(axes[1], aucpr_vs_memory, 'Memory Number Scale Factor', '', 'C1')
        _plot(axes[2], aucpr_vs_depth, 'Depth', '', 'C3')
        _plot(axes[3], aucpr_vs_temperature, 'Temperature', '', 'C2')

        plt.tight_layout()
        sns.despine(fig)

        save_dir = 'results_analysis_paper/ablation_study/'
        
        png_path = os.path.join(save_dir, f'hp_sensitivity_1x4.png')
        pdf_path = os.path.join(save_dir, f'hp_sensitivity_1x4.pdf')

        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.show()

        print(f"Plot saved into {png_path}")
        plt.show()

    plot_hp_sen_1x4()

def plot_contam():
    pima = {
        'MCM': [0.6250, 0.6200, 0.6143, 0.6084],
        'DRL': [0.6322,  0.6278,  0.6225,  0.6139],
        'Disent': [0.6759, 0.6716, 0.6147, 0.6145],
        'LATTE': [0.6986, 0.6933, 0.6798, 0.6764],
    }
    arrhythmia = {
        'MCM': [0.5945, 0.5927, 0.5810, 0.5058], 
        'DRL': [0.5401, 0.5005, 0.4869, 0.3803], 
        'Disent': [0.5953, 0.5932, 0.5842, 0.5105], 
        'LATTE': [0.6113, 0.6091, 0.6010, 0.5330], 
    }

    contamination_rates = [0, 0.01, 0.03, 0.05]
    
    datasets = {'PIMA': pima, 'Arrhythmia': arrhythmia}
    styles = {
        'MCM': {'marker': 'o', 'linestyle': '-'},
        'DRL': {'marker': 's', 'linestyle': '--'},
        'Disent': {'marker': '^', 'linestyle': '-.'},
        'LATTE': {'marker': 'D', 'linestyle': ':'}
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4),)

    for i, (name, data) in enumerate(datasets.items()):
        ax = axes[i]
        for model, values in data.items():
            x1_positions = range(len(contamination_rates)) 
            ax.plot(x1_positions, values, label=model, **styles.get(model, {}))
        
        ax.set_title(name)
        ax.set_xlabel('Contamination Rate (%)')
        ax.set_xticks(x1_positions) 
        ax.set_xticklabels(contamination_rates)      
        ax.legend()
        ax.grid(True, linestyle='--', linewidth=0.5)

        # if i == 0:
        #     ax.set_ylabel('AUC-PR')
    
    plt.tight_layout()
    sns.despine(fig)
    
    save_dir = 'metrics'
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, 'contamination.png')
    pdf_path = os.path.join(save_dir, 'contamination.pdf')

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.show()

    print(f"Plot saved into {png_path}")


def plot_train_ratio():
    arrhythmia = { # from ratio 1, 0.8, 0.5, 0.3 
        'ours': [0.6113, 0.6183, 0.6194, 0.6194, 0.6150], 
        'MCM': [0.5945, 0.5469, 0.5223, 0.5252, 0.5819], 
        'DRL': [0.5401, 0.5469, 0.5945, 0.6025, 0.5819], 
        'Disent': [0.5953, 0.5973, 0.5945, 0.6025, 0.5822],
    }
    pima = { # from ratio 1, 0.8, 0.5 
        'ours': [0.6986, 0.6963, 0.6823, 0.7004, 0.6849], 
        'MCM': [0.6250, 0.6288, 0.6435, 0.6309, 0.6301], 
        'DRL': [0.6322, 0.6340, 0.6440, 0.6622, 0.6843], # temp
        'Disent': [0.6759, 0.6726, 0.6694, 0.6574, 0.6531],
    }
    breastw = {
        'MCM': [0.9910, 0.9920, 0.9893, 0.9864, 0.9847],
        'DRL': [0.9779, 0.9767, 0.9749, 0.9766, 0.9769],
        'Disent': [0.9802, 0.9867, 0.9860, 0.9828, 0.9789],
        'ours': [0.9844, 0.9844, 0.9821, 0.9819, 0.9824],
    }
    shuttle = {
        'ours': [0.9889, 0.9856, 0.9843, 0.9786, 0.9745],
        'MCM': [0.9798, 0.9740, 0.9740, 0.9695, 0.9573],
        'DRL': [0.9893, 0.9840, 0.9907, 0.9843, 0.9826],
        'Disent': [0.9703, 0.970, 0.9704, 0.9670, 0.9641],
    }


    nslkdd = {
        'ours':   [0.9755, 0.9714, 0.9736, 0.9718, 0.9631],
        'MCM':    [0.9792, 0.9778, 0.9764, 0.9739, 0.9673],
        'DRL':    [0.9631, 0.9686, 0.9658, 0.9617, 0.9630],
        'Disent': [0.8466, 0.8587, 0.8442, 0.8556, 0.7292],
    }
    pendigits = { # from ratio 1, 0.8, 0.5 
        'ours': [0.8679, 0.8642, 0.8313, 0.8029, 0.7053], 
        'MCM': [0.8381, 0.8164, 0.8074, 0.7544, 0.6707], 
        'DRL': [0.6094, 0.5, 0.5525, 0.5049, 0.4070], 
        'Disent': [0.7697, 0.7416, 0.6798, 0.6057, 0.4028],
    }

    cardiotocography = { # from ratio 1, 0.8, 0.5 # IDK 
        'ours': [0.6811, 0.6773, 0.6541, 0.6488], 
        'MCM': [0.6344, 0.6271, 0.5541, 0.5623], 
        'DRL': [0.6084, 0.6034, 0.5791, 0.5638], # temp
        'Disent': [0.6856, 0.6910, 0.6920, 0.6895],
    }
    ionosphere = { # from ratio 1, 0.8, 0.5 # IDK 
        'ours': [0.9747, 0.9744, 0.9751, 0.9820], 
        'MCM': [0.8652, 0.7488, 0.5543, 0.7259], 
        'DRL': [0.9412, 0.9207, 0.8910, 0.8576], # temp
        'Disent': [0.9658, 0.9285, 0.9721, 0.9688],
    }        
    satimage = { # from ratio 1, 0.8, 0.5 # IDK 
        'ours': [0.9747, 0.9744, 0.9733, 0.9777, 0.9822], 
        'MCM': [0.8652, 0.7488, 0.6677, 0.5871, 0.9231], 
        'DRL': [0.9412, 0.9207, 0.9289, 0.9271, 0.8800], # temp
        'Disent': [0.9658, 0.9285, 0.9696, 0.9692, 0.9675],
    }
    wbc = {
        'ours': [0.7837, 0.7687, 0.7617, 0.7307, 0.6904],
        'MCM': [0.5548, 0.5823, 0.5751, 0.5635, 0.5298], 
        'DRL': [0.7423, 0.7095, 0.7095, 0.6619, 0.6097], # temp
        'Disent': [0.7566, 0.7461, 0.7533, 0.7169, 0.6842],
    }

    optdigits = {
        'ours': [0.2204, 0.2171, 0.2224, 0.2023, 0.2191],
        'MCM': [0.3372, 0.3284, 0.3286, 0.3208, 0.2878],
        'DRL': [0.2727, 0.2437, 0.2096, 0.2220, 0.1701],
        'Disent': [0.1417, 0.1110, 0.0879, 0.0722, 0.0637],
    }

    trainset_ratio = [1.0, 0.8, 0.6, 0.4, 0.2]
    # arrhythmia, 
    # pima, 
    # pendigits, 
    # satimage-2, 
    # wbc,  
    datasets = {
        'arrhythmia': arrhythmia, 
        'pendigits': pendigits, 
        # 'cardiotocography': cardiotocography,
        'breastw': breastw,
        'shuttle': shuttle, 
        'pima': pima, # DRL increasings
        'nslkdd': nslkdd, # DRL increasings
        'satimage': satimage,
        'wbc': wbc,
        'optdigits': optdigits,
    }
    styles = {
        'MCM': {'marker': 'o', 'linestyle': '-'},
        'DRL': {'marker': 's', 'linestyle': '--'},
        'Disent': {'marker': '^', 'linestyle': '-.'},
        'ours': {'marker': 'D', 'linestyle': ':'}
    }
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(4*len(datasets), 4))

    # Store handles and labels from first subplot for global legend
    legend_handles = []
    legend_labels = []
    
    for i, (name, data) in enumerate(datasets.items()):
        ax = axes[i]
        for model, values in data.items():
            x1_positions = range(len(trainset_ratio)) 
            line = ax.plot(x1_positions, values, label=model, **styles.get(model, {}))
            
            # Collect legend info only from first subplot
            if i == 0:
                legend_handles.extend(line)
                legend_labels.append(model)
        
        ax.set_title(name, fontsize=14)
        ax.set_xlabel('Trainset Ratio (%)', fontsize=12)
        ax.set_xticks(x1_positions) 
        ax.set_xticklabels(trainset_ratio, fontsize=10)
        ax.tick_params(axis='y', labelsize=10)      
        ax.grid(True, linestyle='--', linewidth=0.5)

    fig.legend(legend_handles, legend_labels, loc='upper right', bbox_to_anchor=(0.98, 0.95), ncol=1, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.91) # for legend
    sns.despine(fig)
    
    save_dir = 'results_analysis_paper/unlimited_data/'
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, 'train_ratio.png')
    pdf_path = os.path.join(save_dir, 'train_ratio.pdf')

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.show()

    print(f"Plot saved into {png_path}")


def render_npt():
    npt = {
        'arrhythmia': 0.4345,
        'breastw': 0.8825,
        'campaign': 0.469,
        'cardio': 0.6929,
        'cardiotocography': 0.5334,
        'census': 0.1794, 
        'fraud': 0.6214,
        'glass': 0.2151,
        'ionosphere': 0.9683,
        'mammography': 0.3985,
        'optdigits': 0.1534,
        'pendigits': 0.4844,
        'pima': 0.6798,
        'satellite': 0.8479,
        'satimage-2': 0.6798,
        'shuttle': 0.9403,
        'thyroid': 0.8167,
        'wbc': 0.3997,
        'wine': 0.9985,
    }
    ours = {
         'arrhythmia': 0.6113,
        'breastw': 0.9844,
        'campaign': 0.5105,
        'cardio': 0.8442,
        'cardiotocography': 0.6811,
        'census': 0.2474, 
        'fraud': 0.7240,
        'glass': 0.2909,
        'ionosphere': 0.9772,
        'mammography': 0.4196,
        # 'nslkdd': 0.9755,
        'optdigits': 0.2204,
        'pendigits': 0.8679,
        'pima': 0.6986,
        'satellite-2': 0.8657,
        'satimage': 0.9747,
        'shuttle': 0.9889,
        'thyroid': 0.7566,
        'wbc': 0.7837,
        'wine': 0.8260,
    } 
    # list(npt.values)
    npt_mean = np.mean(list(npt.values()))  # 
    ours_mean = np.mean(list(ours.values()))  # 
    print(f"NPT Mean: {npt_mean:.4f}")
    print(f"Ours Mean: {ours_mean:.4f}")

def render_main_table():
    keys = [
        'ratio_1.0_AUCROC',
        'ratio_1.0_AUCPR',
    ]
    models=  [
        'IForest', 'LOF', 'OCSVM', 'ECOD', 'KNN', 'PCA',  # KNN: 0.6918, LOF: 0.6612
        # 'AutoEncoder', 
        'DeepSVDD', 'GOAD', 
        'NeuTraL', 'ICL', 'MCM', 'DRL',
        'Disent',
    ]

    data = [
        'arrhythmia', 'breastw', 'cardio', 'cardiotocography', 'glass',
        'ionosphere', 'pima', 'wbc', 'wine', 'thyroid', 'optdigits', 'pendigits', 'satellite', 
        'campaign', 'mammography', 'satimage-2', 'nslkdd', 'fraud', 'shuttle', 'census',
    ]
    data.sort()

    my_models = [
        'MemPAE-ws-pos_query+token-d64-lr0.001-t0.1', # this is final
    ]
    results = collect_results()
    df_all, dfs = convert_results_to_csv(results, save_csv=False)
    pivots = make_pivots(dfs, save_csv=False)
    
    for base in keys:
        render(pivots, data, models, my_models, base, 
               add_avg_rank=True, use_rank=False, use_std=False, 
               use_baseline_pr=False, is_temp_tune=False, is_sort=False, is_plot=True)

    


if __name__ == '__main__':
    plot_hp_sen()
    plot_contam()
    plot_train_ratio()
    render_npt()
    # render_main_table()