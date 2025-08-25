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

class MyDataset(Dataset):
    def __init__(self, data, label):
        super(MyDataset, self).__init__()
        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(label)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)


def load_dataset(data_dir, dataset_name, data_dim):
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
    else:
        x = []
        labels = []
        path = os.path.join(data_dir, dataset_name+'.csv')
        with (open(path, 'r')) as data_from:
            csv_reader = csv.reader(data_from)
            for i in csv_reader:
                x.append(i[0:data_dim])
                labels.append(i[data_dim])

        for i in range(len(x)):
            for j in range(data_dim):
                x[i][j] = float(x[i][j])
        for i in range(len(labels)):
            labels[i] = float(labels[i])

        data = np.array(x)
        target = np.array(labels)
        inlier_indices = np.where(target == 0)[0]
        outlier_indices = np.where(target == 1)[0]

        inliers = data[inlier_indices]
        outliers = data[outlier_indices]

    return inliers, outliers


def split_and_preprocess(inliers, outliers, preprocess, ratio=1.0):
    """
    Shuffle, split, and apply preprocessor. 
    """

    # shuffle and split
    np.random.shuffle(inliers)
    num_split = len(inliers) // 2
    train_data = inliers[:num_split]
    train_label = np.zeros(num_split)
    test_data = np.concatenate([inliers[num_split:], outliers], 0)
    test_label = np.zeros(test_data.shape[0])
    test_label[-len(outliers):] = 1


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

def load_and_preprocess(data_dir, dataset_name, data_dim, preprocess):
    inliers, outliers = load_dataset(data_dir, dataset_name, data_dim)
    train_data, train_label, test_data, test_label = split_and_preprocess(inliers, outliers, preprocess, ratio=1.0)

    return train_data, train_label, test_data, test_label