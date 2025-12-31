import torch
from torch.utils.data import DataLoader
from DataSet.MyDataset import MyDataset, MQDataset, DisentDataset, load_and_preprocess
from utils import get_input_dim
dataset_name_list = [
    "arrhythmia", "breastw", "campaign", "cardio", "cardiotocography", 
    "census", "fraud", "glass", "ionosphere", "mammography", 
    "nslkdd", "optdigits", "pendigits", "pima", "satellite", 
    "satimage-2", "shuttle", "thyroid", "wbc", "wine",
]
num_features_list = [
    274, 9, 62, 21, 21,
    500, 29, 9, 33, 6,
    122, 64, 16, 8, 36,
    36, 9, 6, 30, 13,
]

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

def get_multitask_dataloader(train_config: dict):

    contamination_ratio = train_config['contamination_ratio'] if 'contamination_ratio' in train_config else None
    train_ratio =  train_config['train_ratio']
    train_loader_list, test_loader_list = [], []
    for dataset_name in dataset_name_list:
        train_data, train_label, test_data, test_label = load_and_preprocess(train_config['data_dir'], dataset_name, train_config['preprocess'], contamination_ratio, train_ratio)
        train_set = MyDataset(train_data, train_label)
        test_set = MyDataset(test_data, test_label)
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

        train_loader_list.append(train_loader)
        test_loader_list.append(test_loader)

    return train_loader_list, test_loader_list, num_features_list


def get_mq_dataset(train_config: dict):
    """
    Get MQ-specific datasets that include sample indices.
    """
    dataset_name = train_config['dataset_name']
    contamination_ratio = train_config.get('contamination_ratio', None)
    train_ratio = train_config['train_ratio']
    
    train_data, train_label, test_data, test_label = load_and_preprocess(
        train_config['data_dir'], 
        dataset_name, 
        train_config['preprocess'], 
        contamination_ratio, 
        train_ratio
    )
    
    train_set = MQDataset(train_data, train_label)
    test_set = MQDataset(test_data, test_label)
    
    return train_set, test_set


def get_mq_dataloader(train_config: dict):
    """
    Get MQ-specific dataloaders.
    
    Key features:
    1. Returns sample indices for self-exclusion during queue retrieval
    2. Provides num_train for proper queue sizing
    """
    train_set, test_set = get_mq_dataset(train_config)
    
    # Important: shuffle=False for training to maintain consistent index mapping
    # The model handles randomness internally via queue operations
    train_loader = DataLoader(
        train_set,
        batch_size=train_config['batch_size'],
        num_workers=train_config.get('num_workers', 0),
        shuffle=True,  # Can shuffle since we track indices
        pin_memory=True,
        drop_last=False  # Keep all samples to fill queue properly
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=train_config['batch_size'],
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, test_loader

