from torch.utils.data import DataLoader
from DataSet.MyDataset import MyDataset, load_and_preprocess
import os
import glob

def get_dataset(model_config: dict):
    dataset_name = model_config['dataset_name']
    train_data, train_label, test_data, test_label = load_and_preprocess(model_config['data_dir'], dataset_name, model_config['data_dim'], model_config['preprocess'])
    train_set = MyDataset(train_data, train_label)
    test_set = MyDataset(test_data, test_label)

    return train_set, test_set

def get_dataloader(model_config: dict):
    train_set, test_set = get_dataset(model_config)
    train_loader = DataLoader(train_set, 
                              batch_size=model_config['batch_size'], 
                              num_workers=model_config['num_workers'],
                              shuffle=False)
    test_loader = DataLoader(test_set, batch_size=model_config['batch_size'], shuffle=False)

    return train_loader, test_loader