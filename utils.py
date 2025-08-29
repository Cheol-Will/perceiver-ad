#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import numpy as np
import glob
import os
from scipy import io
import yaml
import importlib
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

BASELINE_MODELS = ['OCSVM', 'KNN', 'IForest', 'LOF', 'PCA', 'ECOD', 
                   'DeepSVDD', 'AutoEncoder', 'GOAD', 'ICL', 'NeuTraL']

npz_files = glob.glob(os.path.join('./Data', '*.npz'))
npz_datanames = [os.path.splitext(os.path.basename(file))[0] for file in npz_files]
mat_files = glob.glob(os.path.join('./Data', '*.mat'))
mat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in mat_files]

def load_yaml(args):    
    dict_to_import = args.model_type + '.yaml'
    if args.model_type in BASELINE_MODELS:
        dict_to_import = 'CLASSIC.yaml'

    with open(f'configs/{dict_to_import}', 'r') as f:
        model_configs = yaml.safe_load(f)
    model_config = model_configs['default']
    
    if args.dataname in model_configs:
        for k, v in model_configs[args.dataname].items():
            model_config[k] = v

    if args.model_type in ['Perceiver', 'RIN', 'PAE', 'MemPAE']:
        model_config = replace_transformer_config(args, model_config)
    elif args.model_type in ['MemAE', 'MultiMemAE', 'RINMLP']:
        model_config = replace_mlp_config(args, model_config)

    model_config['model_type'] = args.model_type
    model_config['data_dim'] = get_input_dim(args, model_config)
    model_config['model_type'] = args.model_type
    model_config['dataset_name'] = args.dataname
    model_config['train_ratio'] = args.train_ratio
    model_config['base_path'] = args.base_path    

    return model_config


def build_trainer(model_config):
    model_type = model_config['model_type']
    if model_type == 'DRL':
        from models.DRL.Trainer import Trainer
    elif model_type == 'MCM':
        from models.MCM.Trainer import Trainer
    elif model_type == 'Disent':
        from models.Disent.Trainer import Trainer
    elif model_type == 'Perceiver':
        from models.Perceiver.Trainer import Trainer
    elif model_type == 'RIN':
        from models.RIN.Trainer import Trainer
    elif model_type == 'MemAE':
        from models.MemAE.Trainer import Trainer
    elif model_type == 'MemPAE':
        from models.MemPAE.Trainer import Trainer        
    elif model_type == 'MultiMemAE':
        from models.MultiMemAE.Trainer import Trainer
    elif model_type == 'RINMLP':
        from models.RINMLP.Trainer import Trainer        
    elif model_type == 'PAE':
        from models.PAE.Trainer import Trainer        
    elif model_type in BASELINE_MODELS:
        from models.Baselines.Trainer import Trainer
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return Trainer(model_config)

def get_input_dim(args, model_config):
    if args.dataname in npz_datanames:
        path = os.path.join(model_config['data_dir'], args.dataname + '.npz')
        data = np.load(path)
    elif args.dataname in mat_datanames:
        path = os.path.join(model_config['data_dir'], args.dataname + '.mat')
        data = io.loadmat(path)
    else:
        raise ValueError(f"Unknown dataset {args.dataname}")
    samples = data['X']
    return samples.shape[-1]

def replace_transformer_config(args, model_config):
    model_config['num_heads'] = args.num_heads if args.num_heads is not None else model_config['num_heads']
    model_config['depth'] = args.depth if args.depth is not None else model_config['depth']
    model_config['hidden_dim'] = args.hidden_dim if args.hidden_dim is not None else model_config['hidden_dim']
    model_config['mlp_ratio'] = args.mlp_ratio if args.mlp_ratio is not None else model_config['mlp_ratio']
    model_config['dropout_prob'] = args.dropout_prob if args.dropout_prob is not None else model_config['dropout_prob']
    model_config['learning_rate'] = args.learning_rate if args.learning_rate is not None else model_config['drop_col_prob']

    if args.model_type in ['Perceiver', 'RIN']:
        model_config['drop_col_prob'] = args.drop_col_prob if args.drop_col_prob is not None else model_config['drop_col_prob']
    
    if args.model_type in ['PAE', 'MemPAE']:
        model_config['is_weight_sharing'] = args.is_weight_sharing # default False
        model_config['use_pos_enc_as_query'] = args.use_pos_enc_as_query # default False
        
        if args.model_type == 'MemPAE': 
            model_config['sim_type'] = args.sim_type if args.sim_type is not None else model_config['sim_type']
            model_config['temperature'] = args.temperature if args.temperature is not None else model_config['temperature']

    return model_config

def replace_mlp_config(args, model_config):
    model_config['depth'] = args.depth if args.depth is not None else model_config['depth']
    model_config['hidden_dim'] = args.hidden_dim if args.hidden_dim is not None else model_config['hidden_dim']
    model_config['learning_rate'] = args.learning_rate if args.learning_rate is not None else model_config['drop_col_prob']

    if args.model_type in ['MemAE', 'MultiMemAE']:
        model_config['sim_type'] = args.sim_type if args.sim_type is not None else model_config['sim_type']
        model_config['temperature'] = args.temperature if args.temperature is not None else model_config['temperature']
        if args.model_type == 'MultiMemAE':
            model_config['num_adapters'] = args.num_adapters if args.num_adapters is not None else model_config['num_adapters']

    if args.model_type == 'RINMLP':
        model_config['num_repeat'] = args.num_repeat if args.num_repeat is not None else model_config['num_repeat']

    return model_config


def aucPerformance(score, labels):
    roc_auc = roc_auc_score(labels, score)
    ap = average_precision_score(labels, score)
    return roc_auc, ap

def F1Performance(score, target):
    normal_ratio = (target == 0).sum() / len(target)
    score = np.squeeze(score)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(score))
    pred[score > threshold] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='binary')
    return f1

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger