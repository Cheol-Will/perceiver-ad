import torch
import numpy as np
import argparse
import os
import json

from DataSet.DataLoader import get_dataset


def main(args):
    # dataset_name = 'house'
    # dataset_name = 'california'
    dataset_name = 'wine_quality'
    data_dir = f'Data/{dataset_name}/'
    
    data = {
        'X_num_train': None,
        'X_num_val': None,
        'X_num_test': None,
        'Y_train': None,
        'Y_val': None,
        'Y_test': None,
    }

    for key in data:
        path = os.path.join(data_dir, key + '.npy')
        data[key] = np.load(path)

    # merge X, y
    X = np.concatenate([data['X_num_train'], data['X_num_val'], data['X_num_test']], axis=0)
    y = np.concatenate([data['Y_train'], data['Y_val'], data['Y_test']], axis=0).reshape(-1, 1)
    X = np.concatenate([X, y], axis = 1) # put previous label at the last column.

    # consider as noraml
    y = np.zeros(X.shape[0], dtype=int)

    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    # save
    filepath = 'Data'
    path = os.path.join(filepath, dataset_name+'.npz')

    print(f"Supervised data is saved into {path}")
    np.savez_compressed(path, X=X, y=y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
