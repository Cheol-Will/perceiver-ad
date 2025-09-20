import os, json, glob
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 
import seaborn as sns
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

def plot_hp_sensitivity_old():
    meta_data = {
        'arrhythmia': { # good
            "num_latents": 16, 
            "num_memories": 8, 
            "num_features": 279,
            "num_trainset": 193,
        },
        'cardio': { # good
            "num_latents": 4, 
            "num_memories": 16, 
            "num_features": 21,
            "num_trainset": 827,
        },
        'campaign': { # good
            "num_latents": 4, 
            "num_memories": 128, 
            "num_features": 62,
            "num_trainset": 18274,
        },
        'ionosphere': {
            "num_latents": 4, 
            "num_memories": 8,
            "num_features": 33,
            "num_trainset": 112,
        },
        'mammography': {
            "num_latents": 2, 
            "num_memories": 64,
            "num_features": 6,
            "num_trainset": 5461,
        },
        'optdigits': {
            "num_latents": 8, 
            "num_memories": 32,
            "num_features": 64,
            "num_trainset": 2533,
        },
        'pima': {
            "num_latents": 2, 
            "num_memories": 8,
            "num_features": 8,
            "num_trainset": 250,
        },
        'satimage-2': {
            "num_latents": 4, 
            "num_memories": 32,
            "num_features": 36,
            "num_trainset": 2866,
        },
        'wbc': {
            "num_latents": 4, 
            "num_memories": 8,
            "num_features": 30,
            "num_trainset": 178,
        },
        'wine': {
            "num_latents": 2, 
            "num_memories": 4,
            "num_features": 13,
            "num_trainset": 59,
        },

    }
    aucpr_vs_latent = {
        'arrhythmia': [[8, 0.6087], [16, 0.6113], [32, 0.6134], [64, 0.6119]],
        'cardio': [[2, 0.8371], [4, 0.8442], [8, 0.8320], [16, 0.8350]],
        'campaign': [[2, 0.5013], [4, 0.5105], [8, 0.5077], [16, 0.5159]],
        'ionosphere': [[2, 0.9759], [4, 0.9772], [8, 0.9780], [16, 0.9766]],
        'mammography': [[1, 0.4263], [2, 0.4196], [4, 0.4261], [8, 0.4182]],
        'optdigits': [[4, 0.2147], [8, 0.2204], [16, 0.1752], [32, 0.1593]],
        'pima': [[1, 0.6836], [2, 0.6986], [4, 0.6897], [8, 0.7011]],
        'satimage-2': [[2, 0.9766], [4, 0.9747], [8, 0.9766], [16, 0.9752]],
        'wbc': [[2, 0.7660], [4, 0.7837], [8, 0.7763], [16, 0.7831]],
        'wine': [[1, 0.7932], [2, 0.8260], [4, 0.7728], [8, 0.8526]],
    }
    aucpr_vs_memory = {
        'arrhythmia': [[4, 0.6124], [8, 0.6113], [16, 0.6089], [32, 0.6045]],
        'cardio': [[8, 0.8166], [16, 0.8442], [32, 0.8410], [64, 0.8408]],
        'campaign': [[64, 0.5076], [128, 0.5105], [256, 0.5040], [512, 0.5057]],
        'ionosphere': [[4, 0.9785], [8, 0.9772], [16, 0.9769], [32, 0.9791]],
        'mammography': [[32, 0.3990], [64, 0.4196], [128, 0.4096], [256, 0.4287]],
        'optdigits': [[16, 0.2182], [32, 0.2204], [64, 0.1932], [128, 0.1724]],
        'pima': [[4, 0.6922], [8, 0.6986], [16, 0.7015], [32, 0.6911]],
        'satimage-2': [[16, 0.9741], [32, 0.9747], [64, 0.9741], [128, 0.9742]],
        'wbc': [[4, 0.7824], [8, 0.7837], [16, 0.7862], [32, 0.7809]],
        'wine': [[2, 0.7689], [4, 0.8260], [8, 0.8506], [16, 0.7947]],
    }
    datasets = ['cardio', 'optdigits', 'pima', 'wbc']
    # datasets = ['arrhythmia', 'cardio', 'campaign', 'ionosphere', 'mammography', 'optdigits', 'pima', 'satimage-2', 'wbc', 'wine']
    fig, axes = plt.subplots(2, len(datasets), figsize=(4*len(datasets), 6))

    for i, db in enumerate(datasets):
        ax = axes[0, i]
        data = aucpr_vs_latent[db]
        x, y = zip(*data)
        
        ax.plot(x, y, marker='o', linestyle='-', label='AUCPR')
        default_latent = meta_data[db]['num_features'] ** 0.5
        ax.axvline(x=default_latent, color='C3', linestyle='--', label=f'Default ({default_latent})')
        
        ax.set_xscale('log')
        ax.set_xticks(x)
        
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        
        ax.set_title(f'Dataset: {db.capitalize()}', fontsize=12)
        if i == 0:
             ax.set_ylabel('AUCPR', fontsize=11)
        ax.set_xlabel('Number of Latents', fontsize=11)
        # ax.legend()

    for i, db in enumerate(datasets):
        ax = axes[1, i]
        data = aucpr_vs_memory[db]
        x, y = zip(*data)

        ax.plot(x, y, marker='s', linestyle='-', color='C2', label='AUCPR')
        default_memory = meta_data[db]['num_trainset'] ** 0.5
        ax.axvline(x=default_memory, color='C3', linestyle='--', label=f'Default ({default_memory})')
        
        ax.set_xscale('log')
        ax.set_xticks(x)
        
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        
        if i == 0:
             ax.set_ylabel('AUCPR', fontsize=11)
        ax.set_xlabel('Number of Memories', fontsize=11)
        # ax.legend()
    
    sns.despine(fig)
    plt.tight_layout()
    
    # png_path = 'metrics/hp_sensitivity_all.png'
    # pdf_path = 'metrics/hp_sensitivity_all.pdf'
    png_path = 'metrics/hp_sensitivity.png'
    pdf_path = 'metrics/hp_sensitivity.pdf'
    os.makedirs(os.path.dirname(png_path), exist_ok=True) 
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"plot saved into {png_path}")



def plot_hp_sen():
    aucpr_vs_latent = [
        [0.5, 0.7019],
        [1.0, 0.7124],
        [2.0, 0.7040],
        [4.0, 0.7085],
    ]
    aucpr_vs_memory = [
        [0.5, 0.7044],
        [1.0, 0.7124],
        [2.0, 0.7096],
        [4.0, 0.7071],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    ax1 = axes[0]
    x1, y1 = zip(*aucpr_vs_latent)
    x1_positions = range(len(x1)) 

    ax1.plot(x1_positions, y1, marker='o', linestyle='-', label='Average AUC-PR')

    ax1.set_xticks(x1_positions) 
    ax1.set_xticklabels(x1)      
    ax1.tick_params(axis='x', length=0)

    # ax1.set_title('Latent Number vs AUCPR', fontsize=12)
    ax1.set_ylabel('AUCPR', fontsize=11)
    ax1.set_xlabel('Latent Number Scale Factor', fontsize=11)
    # ax1.legend()
    ax1.grid(True, axis='y', linestyle='--', linewidth=0.5) # 

    ax2 = axes[1]
    x2, y2 = zip(*aucpr_vs_memory)
    x2_positions = range(len(x2))

    ax2.plot(x2_positions, y2, marker='s', linestyle='-', color='C1', label='Average AUC-PR')

    ax2.set_xticks(x2_positions)
    ax2.set_xticklabels(x2)

    ax2.tick_params(axis='x', length=0)

    # ax2.set_title('Memory Number vs AUCPR', fontsize=12)
    ax2.set_xlabel('Memory Number Scale Factor', fontsize=11)
    # ax2.legend()
    ax2.grid(True, axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    sns.despine(fig)

    save_dir = 'metrics'
    os.makedirs(save_dir, exist_ok=True)
    
    png_path = os.path.join(save_dir, 'hp_sensitivity.png')
    pdf_path = os.path.join(save_dir, 'hp_sensitivity.pdf')

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.show()

    print(f"Plot saved into {png_path}")


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
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_train_ratio():
    arrhythmia = { # from ratio 1, 0.8, 0.5, 0.3 
        'ours': [0.6113, 0.6183, 0.6173, 0.6285], 
        'MCM': [0.5945, 0.6041, 0.6035, 0.6127], 
        'DRL': [0.5401, 0.5469, 0.5122, 0.5335], # temp
        'Disent': [0.5945, 0.5973, 0.6000, 0.5885],
    }
    pima = { # from ratio 1, 0.8, 0.5 
        'ours': [0.6986, 0.6963, 0.7021, 0.6820], 
        'MCM': [0.6250, 0.6288, 0.6494, 0.6416], 
        'DRL': [0.6322, 0.6340, 0.6543, 0.6681], # temp
        'Disent': [0.6759, 0.6726, 0.6612, 0.6517],
    }

    pendigits = { # from ratio 1, 0.8, 0.5 
        'ours': [0.8679, 0.8642, 0.8252, 0.7521], 
        'MCM': [0.8381, 0.8164, 0.7561, 0.7010], 
        'DRL': [0.6094, 0.5, 0.4400, 0.4291], # temp
        'Disent': [0.7697, 0.7416, 0.6474, 0.5349],
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
        'ours': [0.9747, 0.9744, 0.9751, 0.9820], 
        'MCM': [0.8652, 0.7488, 0.5543, 0.7259], 
        'DRL': [0.9412, 0.9207, 0.8910, 0.8576], # temp
        'Disent': [0.9658, 0.9285, 0.9721, 0.9688],
    }
    wbc = {
        'ours': [0.7837, 0.7687, 0.7547, 0.7096],
        'MCM': [0.5548, 0.5823, 0.6207, 0.6000], 
        'DRL': [0.7423, 0.7095, 0.6749, 0.6565], # temp
        'Disent': [0.7566, 0.7461, 0.7511, 0.6975],
    }
    trainset_ratio = [1.0, 0.8, 0.5, 0.3]
    # arrhythmia, 
    # pima, 
    # pendigits, 
    # satimage-2, 
    # wbc,  
    datasets = {
        # 'arrhythmia': arrhythmia, 
        'pendigits': pendigits, 
        # 'cardiotocography': cardiotocography, 
        'pima': pima,
        'satimage': satimage,
        'wbc': wbc,
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
        # if i == 0:
        #     ax.set_ylabel('AUC-PR')
    
    # Add single legend for entire figure
    fig.legend(legend_handles, legend_labels, loc='upper right', bbox_to_anchor=(0.98, 0.95), ncol=2, fontsize=12)
    
    plt.tight_layout()
    sns.despine(fig)
    
    save_dir = 'metrics'
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



if __name__ == '__main__':
    plot_hp_sen()
    plot_contam()
    plot_train_ratio()
    render_npt()