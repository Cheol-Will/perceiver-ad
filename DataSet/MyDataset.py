import os
import csv
import numpy as np
import pandas as pd
import glob
from scipy import io
import torch
from torch.utils.data import Dataset
import sklearn.preprocessing
import ipdb


npz_files = glob.glob(os.path.join('./Data', '*.npz'))
npz_datanames = [os.path.splitext(os.path.basename(file))[0] for file in npz_files]

mat_files = glob.glob(os.path.join('./Data', '*.mat'))
mat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in mat_files]

dat_files = glob.glob(os.path.join('./Data', '*.data'))
dat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in dat_files]

arff_files = glob.glob(os.path.join('./Data', '*.arff'))
arff_datanames = [os.path.splitext(os.path.basename(file))[0] for file in arff_files]


class MyDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(label)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)
    
class MQDataset(Dataset):
    """
    Dataset for MQ model that returns sample indices along with data.
    This enables the model to exclude self-samples from the queue during training.
    """
    def __init__(self, data, label):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.long)
        self.num_samples = len(data)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Returns: (data, label, global_index)
        return self.data[idx], self.label[idx], idx

class DisentDataset(Dataset):
    def __init__(self, data, label, patch_size, overlap):
        super().__init__()
        self.data = torch.Tensor(data)
        self.label = torch.Tensor(label)
        self.patch_size = patch_size
        self.overlap = overlap
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        subset_size = self.patch_size
        overlap = self.overlap
        stride = subset_size - overlap
        data = data.unfold(0, subset_size, stride)

        return data, self.label[idx]

    def __len__(self):
        return len(self.data)

def load_dataset(data_dir, dataset_name):
    if dataset_name in npz_datanames:
        path = os.path.join(data_dir, dataset_name+'.npz')
        data=np.load(path)  
        samples = data['X']
        labels = ((data['y']).astype(int)).reshape(-1)

        inliers = samples[labels == 0]
        outliers = samples[labels == 1]
    elif dataset_name in mat_datanames:
        path = os.path.join(data_dir, dataset_name + '.mat')
        data = io.loadmat(path)
        samples = data['X']
        labels = ((data['y']).astype(int)).reshape(-1)

        inliers = samples[labels == 0]
        outliers = samples[labels == 1]


    # below is for datasets from NPT-AD .
    elif dataset_name in dat_datanames:
        path = os.path.join(data_dir, dataset_name + '.data')
        data = io.loadmat(path)
        samples = data['X']
        labels = ((data['y']).astype(int)).reshape(-1)

        inliers = samples[labels == 0]
        outliers = samples[labels == 1]

    
    elif dataset_name in arff_datanames:
        # for seismic
        path = os.path.join(data_dir, dataset_name + '.arff')
        data, _ = io.arff.loadarff(path)
        data = pd.DataFrame(data)
        samples = pd.get_dummies(data.iloc[:, :-1]).to_numpy()
        labels = data.iloc[:, -1].values
        inliers = samples[labels == b'0']
        outliers = samples[labels == b'1']
        print(samples)
        print(labels)

    
    # else:
    #     x = []
    #     labels = []
    #     path = os.path.join(data_dir, dataset_name+'.csv')
    #     with (open(path, 'r')) as data_from:
    #         csv_reader = csv.reader(data_from)
    #         for i in csv_reader:
    #             x.append(i[0:data_dim])
    #             labels.append(i[data_dim])

    #     for i in range(len(x)):
    #         for j in range(data_dim):
    #             x[i][j] = float(x[i][j])
    #     for i in range(len(labels)):
    #         labels[i] = float(labels[i])

    #     data = np.array(x)
    #     target = np.array(labels)
    #     inlier_indices = np.where(target == 0)[0]
    #     outlier_indices = np.where(target == 1)[0]

    #     inliers = data[inlier_indices]
    #     outliers = data[outlier_indices]

    return inliers, outliers


def split_and_preprocess(inliers, outliers, preprocess, ratio=1.0, contamination_ratio=None):
    """
    Shuffle, split, and apply preprocessor. 
    """

    # shuffle and split
    np.random.shuffle(inliers)
    num_split = len(inliers) // 2

    train_data = inliers[:int(num_split*ratio)] # 
    train_label = np.zeros(int(num_split*ratio))
    test_data = np.concatenate([inliers[num_split:], outliers], 0)
    test_label = np.zeros(test_data.shape[0])
    test_label[-len(outliers):] = 1

    if contamination_ratio is not None:
        # extract test outlier and put it in train_data with label=0
        # following contamination ratio.
        num_inliers_train = len(train_data)
        num_outliers_to_add = int(np.ceil(
            (contamination_ratio * num_inliers_train) / (1 - contamination_ratio)
        ))

        if num_outliers_to_add > len(outliers):
            raise ValueError("Not enough data for contamination.")

        outliers_to_add = outliers[:num_outliers_to_add]
        remaining_outliers = outliers[num_outliers_to_add:]

        train_data = np.concatenate([train_data, outliers_to_add], axis=0)
        train_label = np.zeros(len(train_data))

        test_inliers = inliers[num_split:]
        test_data = np.concatenate([test_inliers, remaining_outliers], axis=0)
        
        test_label = np.zeros(len(test_data))
        if len(remaining_outliers) > 0:
            test_label[-len(remaining_outliers):] = 1
        print(f"Train data is contaminated with ratio={contamination_ratio}")

    # preprocessing
    if preprocess == 'standard':
        processor = sklearn.preprocessing.StandardScaler().fit(train_data)
    elif preprocess == 'minmax':
        processor = sklearn.preprocessing.MinMaxScaler().fit(train_data)
    elif preprocess == 'quantile':
        processor = sklearn.preprocessing.QuantileTransformer().fit(train_data)
    elif preprocess == 'none':
        processor = sklearn.preprocessing.FunctionTransformer().fit(train_data)
    train_data = processor.transform(train_data)
    test_data = processor.transform(test_data)

    return train_data, train_label, test_data, test_label   

def load_and_preprocess(data_dir, dataset_name, preprocess, contamination_ratio=None, ratio=1.0):
    inliers, outliers = load_dataset(data_dir, dataset_name)
    train_data, train_label, test_data, test_label = split_and_preprocess(
        inliers, outliers, preprocess, ratio=ratio, contamination_ratio=contamination_ratio)

    return train_data, train_label, test_data, test_label