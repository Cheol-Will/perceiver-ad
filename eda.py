import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
import umap
from DataSet.DataLoader import get_dataset
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def eda(dataset, dataset_name: str, args, output_dir: str = "results_eda"):
    save_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    
    data = dataset.data.numpy() if hasattr(dataset.data, 'numpy') else np.array(dataset.data)
    labels = dataset.targets.numpy() if hasattr(dataset.targets, 'numpy') else np.array(dataset.targets)
    
    normal_mask = labels == 0
    abnormal_mask = labels == 1
    
    normal_data = data[normal_mask]
    abnormal_data = data[abnormal_mask]
    
    n_features = data.shape[1]
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    print(f"\nEDA for Dataset: {dataset_name}")
    print(f"Total samples: {len(data)}")
    print(f"Normal samples: {len(normal_data)}")
    print(f"Abnormal samples: {len(abnormal_data)}")
    print(f"Number of features: {n_features}\n")
    
    ks_stats = None
    if args.all or args.dist or args.box:
        ks_stats = []
        for i in range(n_features):
            ks_stat, _ = ks_2samp(normal_data[:, i], abnormal_data[:, i])
            ks_stats.append(ks_stat)
        ks_stats = np.array(ks_stats)
    
    if args.all or args.importance:
        print("Generating Feature Importance Plot...")
        plot_feature_importance(data, labels, feature_names, save_dir)
    
    if args.all or args.dist:
        print("Generating Distribution Comparison Plot...")
        plot_distributions(normal_data, abnormal_data, feature_names, ks_stats, save_dir)
    
    if args.all or args.box:
        print("Generating Box Plot Comparison...")
        plot_boxplots(normal_data, abnormal_data, feature_names, ks_stats, save_dir)
    
    if args.all or args.mahal:
        print("Generating Mahalanobis Distance Plot...")
        plot_mahalanobis(normal_data, abnormal_data, save_dir)

    if args.all or args.tsne:
        print("Generating t-SNE Plot...")
        plot_tsne(normal_data, abnormal_data, save_dir)

    if args.all or args.umap:
        print("Generating UMAP Plot...")
        plot_umap(normal_data, abnormal_data, save_dir)
    
    print(f"\nEDA complete! Results saved to: {save_dir}")


def plot_feature_importance(data, labels, feature_names, save_dir):
    mi_scores = mutual_info_classif(data, labels, random_state=42)
    
    normal_data = data[labels == 0]
    abnormal_data = data[labels == 1]
    
    variance_ratios = []
    for i in range(data.shape[1]):
        between_var = ((np.mean(normal_data[:, i]) - np.mean(data[:, i]))**2 * len(normal_data) + 
                       (np.mean(abnormal_data[:, i]) - np.mean(data[:, i]))**2 * len(abnormal_data)) / len(data)
        within_var = (np.var(normal_data[:, i]) * len(normal_data) + 
                      np.var(abnormal_data[:, i]) * len(abnormal_data)) / len(data)
        variance_ratios.append(between_var / (within_var + 1e-8))
    variance_ratios = np.array(variance_ratios)
    
    mi_order = np.argsort(-mi_scores)
    vr_order = np.argsort(-variance_ratios)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(feature_names) * 0.3)))
    
    top_n = min(20, len(feature_names))
    
    top_mi_idx = mi_order[:top_n]
    axes[0].barh(range(top_n), mi_scores[top_mi_idx])
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels([feature_names[i] for i in top_mi_idx])
    axes[0].set_xlabel('Mutual Information Score')
    axes[0].set_title(f'Top {top_n} Features by Mutual Information')
    axes[0].invert_yaxis()
    
    top_vr_idx = vr_order[:top_n]
    axes[1].barh(range(top_n), variance_ratios[top_vr_idx])
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels([feature_names[i] for i in top_vr_idx])
    axes[1].set_xlabel('Variance Ratio')
    axes[1].set_title(f'Top {top_n} Features by Variance Ratio')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_distributions(normal_data, abnormal_data, feature_names, ks_stats, save_dir):
    top_features_idx = np.argsort(-ks_stats)[:min(12, len(feature_names))]
    
    n_cols = 3
    n_rows = (len(top_features_idx) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, feat_idx in enumerate(top_features_idx):
        ax = axes[idx]
        feat_name = feature_names[feat_idx]
        
        ax.hist(normal_data[:, feat_idx], bins=30, alpha=0.6, label='Normal', density=True, color='blue')
        ax.hist(abnormal_data[:, feat_idx], bins=30, alpha=0.6, label='Abnormal', density=True, color='red')
        
        try:
            sns.kdeplot(normal_data[:, feat_idx], ax=ax, color='blue', linewidth=2)
            sns.kdeplot(abnormal_data[:, feat_idx], ax=ax, color='red', linewidth=2)
        except:
            pass
        
        ax.set_title(f'{feat_name}\nKS={ks_stats[feat_idx]:.3f}')
        ax.legend()
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
    
    for idx in range(len(top_features_idx), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Distribution Comparison: Top Discriminative Features', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distribution_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_boxplots(normal_data, abnormal_data, feature_names, ks_stats, save_dir):
    top_features_idx = np.argsort(-ks_stats)[:min(12, len(feature_names))]
    
    n_cols = 3
    n_rows = (len(top_features_idx) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, feat_idx in enumerate(top_features_idx):
        ax = axes[idx]
        feat_name = feature_names[feat_idx]
        
        data_to_plot = [normal_data[:, feat_idx], abnormal_data[:, feat_idx]]
        bp = ax.boxplot(data_to_plot, labels=['Normal', 'Abnormal'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_title(feat_name)
        ax.set_ylabel('Value')
    
    for idx in range(len(top_features_idx), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Box Plot Comparison: Top Discriminative Features', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'boxplot_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_mahalanobis(normal_data, abnormal_data, save_dir):
    normal_mean = np.mean(normal_data, axis=0)
    
    try:
        cov_matrix = np.cov(normal_data.T)
        cov_inv = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6)
        
        def mahalanobis_dist(x, mean, cov_inv):
            diff = x - mean
            return np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
        
        normal_mahal = [mahalanobis_dist(x, normal_mean, cov_inv) for x in normal_data]
        abnormal_mahal = [mahalanobis_dist(x, normal_mean, cov_inv) for x in abnormal_data]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(normal_mahal, bins=50, alpha=0.6, label='Normal', density=True, color='blue')
        ax.hist(abnormal_mahal, bins=50, alpha=0.6, label='Abnormal', density=True, color='red')
        ax.set_xlabel('Mahalanobis Distance')
        ax.set_ylabel('Density')
        ax.set_title('Mahalanobis Distance Distribution')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'mahalanobis_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"  Warning: Could not compute Mahalanobis distances: {e}")


def plot_tsne(normal_data, abnormal_data, save_dir):
    X = np.vstack([normal_data, abnormal_data])
    y = np.concatenate([np.zeros(len(normal_data)), np.ones(len(abnormal_data))])
    
    if len(X) > 3000:
        idx = np.random.choice(len(X), 3000, replace=False)
        X = X[idx]
        y = y[idx]
    
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], c='blue', label='Normal', alpha=0.5, s=15)
    plt.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], c='red', label='Abnormal', alpha=0.5, s=15)
    plt.title("t-SNE Visualization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_umap(normal_data, abnormal_data, save_dir):
    X = np.vstack([normal_data, abnormal_data])
    y = np.concatenate([np.zeros(len(normal_data)), np.ones(len(abnormal_data))])
    
    if len(X) > 3000:
        idx = np.random.choice(len(X), 3000, replace=False)
        X = X[idx]
        y = y[idx]
    
    reducer = umap.UMAP(random_state=42)
    X_embedded = reducer.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], c='blue', label='Normal', alpha=0.5, s=15)
    plt.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], c='red', label='Abnormal', alpha=0.5, s=15)
    plt.title("UMAP Visualization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'umap_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_synthetic_dataset():
    class MockDataset:
        def __init__(self):
            np.random.seed(42)
            n_normal = 800
            n_abnormal = 200
            n_features = 20
            
            normal_data = np.random.randn(n_normal, n_features)
            abnormal_data = np.random.randn(n_abnormal, n_features) * 1.5 + 1.0
            abnormal_data[:50, :3] += 3.0
            abnormal_data[50:100, 5:8] -= 2.5
            
            self.data = np.vstack([normal_data, abnormal_data]).astype(np.float32)
            self.targets = np.concatenate([np.zeros(n_normal), np.ones(n_abnormal)]).astype(np.float32)
    
    return MockDataset()


def main(args):
    dict_to_import = args.model_type + '.yaml'
    with open(f'configs/{dict_to_import}', 'r') as f:
        configs = yaml.safe_load(f)
    train_config = configs['default']['train_config']
    train_config['dataset_name'] = args.dataname
    train_config['train_ratio'] = args.train_ratio
    train_set, test_set = get_dataset(train_config)
    dataset_name = train_config['dataset_name']
    
    eda(test_set, dataset_name, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EDA for Tabular Anomaly Detection')
    parser.add_argument('--dataname', type=str, default='wine', help='Dataset name')
    parser.add_argument('--model_type', type=str, default='DRL')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--train_ratio', type=float, default=1.0)
    
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--importance', action='store_true', help='Feature importance')
    parser.add_argument('--dist', action='store_true', help='Distribution comparison')
    parser.add_argument('--box', action='store_true', help='Boxplot comparison')
    parser.add_argument('--tsne', action='store_true', help='t-SNE visualization')
    parser.add_argument('--umap', action='store_true', help='UMAP visualization')
    parser.add_argument('--mahal', action='store_true', help='Mahalanobis distance')

    args = parser.parse_args()
    
    main(args)