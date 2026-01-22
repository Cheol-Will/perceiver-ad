import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap 

plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'axes.titlesize': 20})
plt.rcParams.update({'axes.labelsize': 18})
plt.rcParams.update({'xtick.labelsize': 16})
plt.rcParams.update({'ytick.labelsize': 16})
plt.rcParams.update({'legend.fontsize': 16})
plt.rcParams.update({'figure.titlesize': 22})

def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

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
    print(f"Histogram plot saved at: {out_path}")
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


def _plot_heatmap_on_ax(ax, data, vmin, vmax, xticklabels, yticklabels, title=None, ylabel=None):
    sns.heatmap(
        data, ax=ax, cmap="viridis", square=True,
        xticklabels=xticklabels, yticklabels=yticklabels,
        vmin=vmin, vmax=vmax, cbar=True
    )
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=20, fontweight='bold')
    if xticklabels:
        ax.set_xticklabels(xticklabels, rotation=90)

def plot_attn_heatmap(train_config, attn_enc, attn_dec, labels, feature_names=None):
    attn_enc = _to_numpy(attn_enc)
    attn_dec = _to_numpy(attn_dec)
    labels = np.array(labels)

    normal_idx = np.where(labels == 'Test-Normal')[0]
    abnormal_idx = np.where(labels == 'Test-Abnormal')[0]

    if len(normal_idx) == 0 or len(abnormal_idx) == 0:
        return

    seq_len = attn_enc.shape[-1]
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(seq_len - 1)]
    tick_labels = ['CLS'] + feature_names

    layers = []
    for i in range(attn_enc.shape[1]):
        layers.append((attn_enc[:, i, :, :, :], f"Encoder L{i}"))
    for i in range(attn_dec.shape[1]):
        layers.append((attn_dec[:, i, :, :, :], f"Decoder L{i}"))

    sample_indices = [normal_idx[0], normal_idx[1], abnormal_idx[0], abnormal_idx[1]]
    row_titles = [
        "Normal-Avg", "Abnormal-Avg",
        "Normal-1", "Normal-2", "Abnormal-1", "Abnormal-2"
    ]

    num_cols = len(layers)
    num_rows = len(row_titles)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    if num_cols == 1:
        axes = axes[:, np.newaxis]

    for col, (layer_data, layer_name) in enumerate(layers):
        attn_avg_head = layer_data.mean(axis=1)

        means = [
            attn_avg_head[normal_idx].mean(axis=0),
            attn_avg_head[abnormal_idx].mean(axis=0)
        ]
        samples = list(attn_avg_head[sample_indices])
        
        all_maps = means + samples
        
        vmin_mean = min(m.min() for m in means)
        vmax_mean = max(m.max() for m in means)
        vmin_samp = min(s.min() for s in samples)
        vmax_samp = max(s.max() for s in samples)

        for row, heatmap_data in enumerate(all_maps):
            ax = axes[row, col]
            is_mean_row = row < 2
            is_first_col = (col == 0)
            is_first_row = (row == 0)
            is_last_row = (row == num_rows - 1)

            current_vmin = vmin_mean if is_mean_row else vmin_samp
            current_vmax = vmax_mean if is_mean_row else vmax_samp
            
            title = None

            if is_first_row:
                # sub_name = "Normal" if row == 0 else "Abnormal"
                title = layer_name
            
            x_ticks = tick_labels if is_last_row else []
            y_ticks = tick_labels if is_first_col else []
            y_label = row_titles[row] if is_first_col else None

            _plot_heatmap_on_ax(
                ax, heatmap_data, current_vmin, current_vmax,
                x_ticks, y_ticks, title, y_label
            )

    dataset_name = train_config.get("dataset_name", "result")
    plt.suptitle(f"Attention Heatmaps: {dataset_name}", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    out_path = os.path.join(train_config['base_path'], "attention_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined attention heatmap: {out_path}")
    plt.close()