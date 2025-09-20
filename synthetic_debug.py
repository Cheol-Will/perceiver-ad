import torch
import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt
from DataSet.DataLoader import get_dataset
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import matplotlib.pyplot as plt
import torch

def save_heatmap(X: np.ndarray, title: str, out_path: str):
    # Note that constant column cannot define correlation coefficients 
    # with other columns; thus, rendered as white line 
    corr_matrix = np.corrcoef(X.T)
    plt.figure(figsize=(10, 8), dpi=200)
    im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='equal', vmin=-1, vmax=1)
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('Feature Index', fontsize=12)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()
    
    print(f"Heatmap saved to: {out_path}")




def save_mutual_info_heatmap(X: np.ndarray, title: str, out_path: str):
    """
    효율적인 Mutual Information 히트맵 생성 함수
    """
    n_features = X.shape[1]
    
    # 한 번에 모든 피처를 이산화
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    X_discrete = discretizer.fit_transform(X).astype(int)
    
    # GPU 사용 가능하면 torch 사용, 아니면 numpy
    if torch.cuda.is_available():
        mi_matrix = compute_mi_torch(X_discrete, n_features)
    else:
        mi_matrix = compute_mi_numpy(X_discrete, n_features)
    
    # 대각선은 1.0으로 설정 (자기 자신과의 MI는 최대)
    np.fill_diagonal(mi_matrix, 1.0)
    
    # 정규화
    mi_matrix_norm = mi_matrix / (mi_matrix.max() + 1e-8)
    
    plt.figure(figsize=(10, 8), dpi=200)
    im = plt.imshow(mi_matrix_norm, cmap='coolwarm', aspect='equal', vmin=0, vmax=1)
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('Feature Index', fontsize=12)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Mutual Information', rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()
    
    print(f"Mutual Information heatmap saved to: {out_path}")


def compute_mi_torch(X_discrete: np.ndarray, n_features: int) -> np.ndarray:
    """
    PyTorch를 사용한 효율적인 MI 계산 (GPU 가속)
    """
    X_tensor = torch.from_numpy(X_discrete).cuda().float()
    mi_matrix = torch.zeros(n_features, n_features, device='cuda')
    
    # 배치 처리로 여러 피처 쌍을 동시에 계산
    batch_size = min(64, n_features)  # 메모리에 맞게 조정
    
    for i in range(0, n_features, batch_size):
        end_i = min(i + batch_size, n_features)
        batch_i = X_tensor[:, i:end_i]
        
        for j in range(n_features):
            if i <= j < end_i:
                # 대칭 행렬이므로 상삼각만 계산
                feature_j = X_tensor[:, j]
                
                for k, idx_i in enumerate(range(i, end_i)):
                    if idx_i != j:
                        mi_val = mutual_info_score_torch(batch_i[:, k], feature_j)
                        mi_matrix[idx_i, j] = mi_val
                        mi_matrix[j, idx_i] = mi_val  # 대칭성 이용
    
    return mi_matrix.cpu().numpy()


def mutual_info_score_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    PyTorch를 사용한 빠른 MI 계산
    """
    # CPU로 이동해서 sklearn 사용 (더 안정적)
    x_cpu = x.cpu().numpy().astype(int)
    y_cpu = y.cpu().numpy().astype(int)
    
    return mutual_info_score(x_cpu, y_cpu)


def compute_mi_numpy(X_discrete: np.ndarray, n_features: int) -> np.ndarray:
    """
    NumPy를 사용한 효율적인 MI 계산
    """
    mi_matrix = np.zeros((n_features, n_features))
    
    # 상삼각 행렬만 계산 (대칭성 이용)
    for i in range(n_features):
        for j in range(i+1, n_features):
            mi_val = mutual_info_score(X_discrete[:, i], X_discrete[:, j])
            mi_matrix[i, j] = mi_val
            mi_matrix[j, i] = mi_val  # 대칭성 이용
    
    return mi_matrix


def main(args):
    data_list = [
        'arrhythmia',
        'breastw', 
        'cardio',
        'cardiotocography',
        'glass',
        'ionosphere', 
        'pima',
        'wbc',
        'wine',
        'thyroid',
        'optdigits',
        'pendigits',
        'satellite',
        'campaign',
        'mammography',
        'nslkdd',
        'shuttle',
        'fraud',
        'census'
    ]

    for dataname in data_list:
            
        model_config = {
            'dataset_name': dataname,
            # 'dataset_name': 'dependency_anomalies_45_wine_42',
            'data_dir': 'Data/',
            'data_dim': 1,
            'preprocess': 'standard',
        }
        
        train_set, test_set = get_dataset(model_config)
        print(f"Train set shape: {train_set.data.shape}")
        print(f"All train targets are 0: {(train_set.targets == 0).all()}")
        print(f"Test set shape: {test_set.data.shape}")
        unique_targets, counts = torch.unique(test_set.targets, return_counts=True)
        print(f"Test set target distribution: {dict(zip(unique_targets.tolist(), counts.tolist()))}")
        
        if hasattr(train_set.data, 'numpy'):
            train_data = train_set.data.numpy()
        else:
            train_data = train_set.data
        
        if hasattr(test_set.data, 'numpy'):
            test_data = test_set.data.numpy()
        else:
            test_data = test_set.data
        
        # save_heatmap(
        #     X=train_data,
        #     title=f"Feature Correlation Matrix - {model_config['dataset_name']} (Train)",
        #     out_path=f"Data/train_correlation_{model_config['dataset_name']}.png"
        # )
        save_mutual_info_heatmap(
            X=train_data, 
            title=f"Mutual Information - {model_config['dataset_name']} (Train)", 
            out_path=f"Data/train_mi_{model_config['dataset_name']}.png"
        )
    # save_heatmap(
    #     X=test_data,
    #     title=f"Feature Correlation Matrix - {model_config['dataset_name']} (Test)",
    #     out_path=f"test_correlation_{model_config['dataset_name']}.png"
    # )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--r')
    
    args = parser.parse_args()
    main(args)