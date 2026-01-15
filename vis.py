import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE 

def plot_latent(train_config, target, label):
    
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    z_embedded = tsne.fit_transform(target)
    
    df_plot = pd.DataFrame(z_embedded, columns=['Dim1', 'Dim2'])
    df_plot['Label'] = label
    colors_dict = {
        'Train-Normal': 'lightgray',
        'Test-Normal': 'blue',
        'Test-Abnormal': 'red',
    }
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_plot, 
        x='Dim1', 
        y='Dim2', 
        hue='Label', 
        style='Label', 
        palette=colors_dict, 
        alpha=0.7 
    )
    
    plt.title(f"t-SNE Visualization - {train_config.get('dataset_name', 'Dataset')}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    base_path = train_config['base_path']
    os.makedirs(base_path, exist_ok=True) 
    
    out_path = os.path.join(base_path, f'latent_tsne_{train_config.get("dataset_name", "result")}.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    print(f"Latent plot saved at: {out_path}")
    
    plt.close()

def plot_score_hist(train_config, target, label, score_name='Score'):
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    df_plot = pd.DataFrame({
        score_name: target,
        'Label': label
    })
    
    colors_dict = {
        'Train-Normal': 'lightgray',
        'Test-Normal': 'blue',
        'Test-Abnormal': 'red',
    }
    
    base_path = train_config['base_path']
    dataset_name = train_config.get("dataset_name", "result")
    os.makedirs(base_path, exist_ok=True) 

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_plot,
        x=score_name,
        hue='Label',
        palette=colors_dict,
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
        alpha=0.5
    )
    plt.title(f"{score_name} Distribution - {dataset_name}")
    plt.xlabel(score_name)
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(base_path, f'hist_{score_name}_{dataset_name}.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    print(f"Original Histogram plot saved at: {out_path}")
    plt.close()

    threshold = np.percentile(target, 95)
    df_filtered = df_plot[df_plot[score_name] <= threshold]

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_filtered,
        x=score_name,
        hue='Label',
        palette=colors_dict,
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
        alpha=0.5
    )
    plt.title(f"{score_name} Distribution (Outlier Removed 0.95) - {dataset_name}")
    plt.xlabel(score_name)
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)

    out_path_removed = os.path.join(base_path, f'hist_{score_name}_{dataset_name}_outlier_removal.png')
    plt.savefig(out_path_removed, bbox_inches='tight', dpi=200)
    print(f"Filtered Histogram plot saved at: {out_path_removed}")
    plt.close()