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
            "num_trainset": 226,
        },
        
        'cardio': { # good
            "num_latents": 4, 
            "num_memories": 16, 
            "num_features": 21,
            "num_trainset": 915,
        },
        'ionosphere': {
            "num_latents": 4, 
            "num_memories": 8,
            "num_features": 33,
            "num_trainset": 175,
        },
        'optdigits': {
            "num_latents": 8, 
            "num_memories": 32,
            "num_features": 64,
            "num_trainset": 2608,
        },
        'pima' : {
            "num_latents": 2, 
            "num_memories": 8,
            "num_features": 8,
            "num_trainset": 384,
        },
        'wbc' : {
            "num_latents": 4, 
            "num_memories": 8,
            "num_features": 30,
            "num_trainset": 139,
        },

    }
    aucpr_vs_memory = {
        'arrhythmia': [[4, 0.6124], [8, 0.6113], [16, 0.6089], [32, 0.6045]],
        'cardio': [[8, 0.8166], [16, 0.8442], [32, 0.8410], [64, 0.8408]],
        'ionosphere': [[4, 0.9785], [8, 0.9772], [16, 0.9769], [32, 0.9791]],
        'optdigits': [[16, 0.2182], [32, 0.2204], [64, 0.1932], [128, 0.1724]],
        'pima': [[4, 0.6922], [8, 0.6986], [16, 0.7015], [32, 0.6911]],
        'wbc': [[4, 0.7824], [8, 0.7837], [16, 0.7862], [32, 0.7809]],
    }
    aucpr_vs_latent = {
        'arrhythmia': [[8, 0.6087], [16, 0.6113], [32, 0.6134], [64, 0.6119]],
        'cardio': [[2, 0.8371], [4, 0.8442], [8, 0.8320], [16, 0.8350]],
        'ionosphere': [[2, 0.9759], [4, 0.9772], [8, 0.9780], [16, 0.9766]],
        'optdigits': [[4, 0.2147], [8, 0.2204], [16, 0.1752], [32, 0.1593]],
        'pima': [[1, 0.6836], [2, 0.6986], [4, 0.6897], [8, 0.7011]],
        'wbc': [[2, 0.7660], [4, 0.7837], [8, 0.7763], [16, 0.7831]],
    }

    datasets = ['cardio', 'optdigits', 'wbc']
    datasets = ['arrhythmia', 'cardio', 'ionosphere', 'optdigits', 'pima', 'wbc']
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
    
    out_path = 'metrics/hp_sensitivity.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True) 
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"plot saved into {out_path}")



if __name__ == '__main__':
    plot_hp_sensitivity()