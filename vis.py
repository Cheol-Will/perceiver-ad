import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap 

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.titlesize': 16})
plt.rcParams.update({'axes.labelsize': 14})
plt.rcParams.update({'xtick.labelsize': 12})
plt.rcParams.update({'ytick.labelsize': 12})
plt.rcParams.update({'legend.fontsize': 12})
plt.rcParams.update({'figure.titlesize': 18})

def plot_tsne(train_config, target, label, target_name='latent'):
    
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
    
    plt.title(f"t-SNE Visualization of {target_name} - {train_config.get('dataset_name', 'Dataset')}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    base_path = train_config['base_path']
    os.makedirs(base_path, exist_ok=True) 
    
    out_path = os.path.join(base_path, f'{target_name}_tsne_{train_config.get("dataset_name", "result")}.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    print(f"{target_name} (t-SNE) plot saved at: {out_path}")
    
    plt.close()

    print(f"Running UMAP on {target_name}...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    u_embedded = reducer.fit_transform(target)

    df_plot_umap = pd.DataFrame(u_embedded, columns=['Dim1', 'Dim2'])
    df_plot_umap['Label'] = label

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_plot_umap, 
        x='Dim1', 
        y='Dim2', 
        hue='Label', 
        style='Label', 
        palette=colors_dict, 
        alpha=0.7 
    )
    
    plt.title(f"UMAP Visualization of {target_name} - {train_config.get('dataset_name', 'Dataset')}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path_umap = os.path.join(base_path, f'{target_name}_umap_{train_config.get("dataset_name", "result")}.png')
    plt.savefig(out_path_umap, bbox_inches='tight', dpi=200)
    print(f"{target_name} (UMAP) plot saved at: {out_path_umap}")
    
    plt.close()


def plot_tsne_input_and_recon(train_config, target1, target2, base_labels, target_name='input_vs_recon'):

    if isinstance(target1, torch.Tensor):
        target1 = target1.detach().cpu().numpy()
    if isinstance(target2, torch.Tensor):
        target2 = target2.detach().cpu().numpy()
    target = np.concatenate([target1, target2], axis=0) 

    labels_input = [f"{l}-Input" for l in base_labels]
    labels_recon = [f"{l}-Recon" for l in base_labels]
    
    final_labels = labels_input + labels_recon
    
    colors_dict = {
        'Train-Normal-Input': 'lightgray', 
        'Test-Normal-Input': 'blue',
        'Test-Abnormal-Input': 'red',

        'Train-Normal-Recon': 'lightgray',       
        'Test-Normal-Recon': 'blue',
        'Test-Abnormal-Recon': 'red',
    }
    
    alpha_dict = {
        'Train-Normal-Input': 0.3, 
        'Test-Normal-Input': 0.3,
        'Test-Abnormal-Input': 0.3,

        'Train-Normal-Recon': 0.8,
        'Test-Normal-Recon': 0.8,
        'Test-Abnormal-Recon': 0.8,
    }
    
    markers_dict = {
        'Train-Normal-Input': 'o', 
        'Test-Normal-Input': 'o',
        'Test-Abnormal-Input': 'o',

        'Train-Normal-Recon': 'X',
        'Test-Normal-Recon': 'X',
        'Test-Abnormal-Recon': 'X',
    }

    base_path = train_config['base_path']
    os.makedirs(base_path, exist_ok=True) 

    print(f"Running t-SNE on shape {target.shape}...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    z_embedded = tsne.fit_transform(target)
    
    df_plot = pd.DataFrame(z_embedded, columns=['Dim1', 'Dim2'])
    df_plot['Label'] = final_labels
    
    plt.figure(figsize=(12, 10))
    
    for label in df_plot['Label'].unique():
        df_subset = df_plot[df_plot['Label'] == label]
        plt.scatter(
            df_subset['Dim1'], 
            df_subset['Dim2'],
            c=colors_dict[label],
            marker=markers_dict[label],
            s=60,
            alpha=alpha_dict[label],
            label=label
        )
    
    plt.title(f"t-SNE Visualization of {target_name} - {train_config.get('dataset_name', 'Dataset')}")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_path = os.path.join(base_path, f'{target_name}_tsne_{train_config.get("dataset_name", "result")}.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    print(f"{target_name} (t-SNE) plot saved at: {out_path}")
    
    plt.close()

    print(f"Running UMAP on shape {target.shape}...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    u_embedded = reducer.fit_transform(target)
    
    df_plot_umap = pd.DataFrame(u_embedded, columns=['Dim1', 'Dim2'])
    df_plot_umap['Label'] = final_labels

    plt.figure(figsize=(12, 10))
    
    for label in df_plot_umap['Label'].unique():
        df_subset = df_plot_umap[df_plot_umap['Label'] == label]
        plt.scatter(
            df_subset['Dim1'], 
            df_subset['Dim2'],
            c=colors_dict[label],
            marker=markers_dict[label],
            s=60,
            alpha=alpha_dict[label],
            label=label
        )

    plt.title(f"UMAP Visualization of {target_name} - {train_config.get('dataset_name', 'Dataset')}")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_path_umap = os.path.join(base_path, f'{target_name}_umap_{train_config.get("dataset_name", "result")}.png')
    plt.savefig(out_path_umap, bbox_inches='tight', dpi=200)
    print(f"{target_name} (UMAP) plot saved at: {out_path_umap}")
    
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



def plot_attn_heatmap(train_config, attn_enc, attn_dec, labels, feature_names=None):
    def to_numpy(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    attn_enc = to_numpy(attn_enc)
    attn_dec = to_numpy(attn_dec)
    labels = np.array(labels)

    seq_len = attn_enc.shape[-1]
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(seq_len - 1)]
    
    tick_labels = ['CLS'] + feature_names
    base_path = train_config['base_path']
    dataset_name = train_config.get("dataset_name", "result")

    normal_idx = np.where(labels == 'Test-Normal')[0]
    abnormal_idx = np.where(labels == 'Test-Abnormal')[0]

    if len(normal_idx) == 0 or len(abnormal_idx) == 0:
        return

    layers_enc = [(attn_enc[:, i, :, :, :], f"Encoder L{i}") for i in range(attn_enc.shape[1])]
    layers_dec = [(attn_dec[:, i, :, :, :], f"Decoder L{i}") for i in range(attn_dec.shape[1])]
    layers = layers_enc + layers_dec
    num_cols = len(layers)

    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10))
    if num_cols == 1: axes = axes[:, np.newaxis]
    
    cmap = "viridis"

    for col, (data, name) in enumerate(layers):
        attn_avg_head = data.mean(axis=1) 
        
        mean_maps = [
            attn_avg_head[normal_idx].mean(axis=0),
            attn_avg_head[abnormal_idx].mean(axis=0)
        ]
        titles = [f"{name} (Normal)", f"{name} (Abnormal)"]
        
        vmin = min(m.min() for m in mean_maps)
        vmax = max(m.max() for m in mean_maps)

        for row, (heatmap_data, title) in enumerate(zip(mean_maps, titles)):
            ax = axes[row, col]
            is_bottom = (row == 1)
            is_left = (col == 0)

            sns.heatmap(heatmap_data, ax=ax, cmap=cmap, square=True,
                        xticklabels=tick_labels if is_bottom else [],
                        yticklabels=tick_labels if is_left else [],
                        vmin=vmin, vmax=vmax, cbar=True)
            
            ax.set_title(title)
            if is_bottom:
                ax.set_xticklabels(tick_labels, rotation=90)

    plt.suptitle(f"Attention Heatmap: {dataset_name}", fontsize=18)
    plt.tight_layout()
    
    out_path = os.path.join(base_path, "attention_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined attention heatmap: {out_path}")

    sample_indices = [normal_idx[0], normal_idx[1], abnormal_idx[0], abnormal_idx[1]]
    row_titles = ["Normal-1", "Normal-2", "Abnormal-1", "Abnormal-2"]
    
    num_rows = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    if num_cols == 1: axes = axes[:, np.newaxis]

    for col, (data, name) in enumerate(layers):
        attn_avg_head = data.mean(axis=1)
        selected_maps = attn_avg_head[sample_indices]
        
        vmin, vmax = selected_maps.min(), selected_maps.max()

        for row in range(num_rows):
            ax = axes[row, col]
            heatmap_data = selected_maps[row]
            
            is_last_row = (row == num_rows - 1)
            is_first_col = (col == 0)

            sns.heatmap(heatmap_data, ax=ax, cmap=cmap, square=True,
                        xticklabels=tick_labels if is_last_row else [],
                        yticklabels=tick_labels if is_first_col else [],
                        vmin=vmin, vmax=vmax, cbar=True)

            if row == 0:
                ax.set_title(name)
            
            if is_first_col:
                ax.set_ylabel(row_titles[row], fontsize=12, fontweight='bold')
            
            if is_last_row:
                ax.set_xticklabels(tick_labels, rotation=90)

    plt.suptitle(f"Individual Attention Heatmaps (4 Samples): {dataset_name}", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    out_path_samples = os.path.join(base_path, "attention_heatmap_samples.png")
    plt.savefig(out_path_samples, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sample-wise attention heatmap: {out_path_samples}")