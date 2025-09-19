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

def plot_hp_sensitivity():
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
        [0.5, 0.7267],
        [1.0, 0.7369],
        [2.0, 0.7284],
        [4.0, 0.7327],
    ]
    aucpr_vs_memory = [
        [0.5, 0.7294],
        [1.0, 0.7369],
        [2.0, 0.7344],
        [4.0, 0.7322],
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

if __name__ == '__main__':
    plot_hp_sen()