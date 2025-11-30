import torch
from torch.utils.data import DataLoader
from DataSet.MyDataset import MyDataset, DisentDataset, load_and_preprocess

def get_dataset(train_config: dict):
    dataset_name = train_config['dataset_name']
    contamination_ratio = train_config['contamination_ratio'] if 'contamination_ratio' in train_config else None
    train_ratio =  train_config['train_ratio']
    train_data, train_label, test_data, test_label = load_and_preprocess(train_config['data_dir'], dataset_name, train_config['preprocess'], contamination_ratio, train_ratio)
    train_set = MyDataset(train_data, train_label)
    test_set = MyDataset(test_data, test_label)

    return train_set, test_set

def get_disent_dataset(train_config: dict):
    dataset_name = train_config['dataset_name']
    contamination_ratio = train_config['contamination_ratio'] if 'contamination_ratio' in train_config else None
    train_ratio =  train_config['train_ratio']
    patch_size = train_config['patch_size']
    overlap = train_config['overlap']
    train_data, train_label, test_data, test_label = load_and_preprocess(train_config['data_dir'], dataset_name, train_config['preprocess'], contamination_ratio, train_ratio)
    train_set = DisentDataset(train_data, train_label, patch_size, overlap)
    test_set = DisentDataset(test_data, test_label, patch_size, overlap)

    return train_set, test_set

def get_dataloader(train_config: dict):
    if train_config['model_type'] == 'Disent': 
        train_set, test_set = get_disent_dataset(train_config)
    else:
        train_set, test_set = get_dataset(train_config)

    
    if train_config['model_type'] == 'NPTAD':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,
            shuffle=True
        )
        train_loader = DataLoader(
            train_set,
            batch_size=train_config['batch_size'],
            sampler=train_sampler,
            num_workers=train_config['num_workers'],
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_set, 
            batch_size=train_config['batch_size'], 
            num_workers=train_config['num_workers'],
            shuffle=True,
            pin_memory=True
        )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=train_config['batch_size'], 
        shuffle=False,
        pin_memory=True
    )

    return train_loader, test_loader