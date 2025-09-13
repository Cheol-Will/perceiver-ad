import torch
import numpy as np
import argparse
import os
import json

from DataSet.DataLoader import get_dataset


def main(args):
    model_config = {
        'dataset_name': 'dependency_anomalies_45_wine_42',
        'data_dir': 'Data/',
        'data_dim': 1,
        'preprocess': 'standard',
    }
    train_set, test_set = get_dataset(model_config)
    print(train_set.data.shape)
    print((train_set.targets == 0).all())

    print(test_set.data.shape)
    print(torch.unique(test_set.targets, return_counts=True))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--r')
    
    args = parser.parse_args()
    main(args)