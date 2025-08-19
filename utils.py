#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import logging
import numpy as np
import glob
import os
from scipy import io
import importlib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
npz_files = glob.glob(os.path.join('./Data', '*.npz'))
npz_datanames = [os.path.splitext(os.path.basename(file))[0] for file in npz_files]

mat_files = glob.glob(os.path.join('./Data', '*.mat'))
mat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in mat_files]

DEEP_MODELS = ['Perceiver', 'MCM', 'DRL']
CLASSIC_MODELS = ['OCSVM', 'KNN', 'IForest', 'LOF', 'PCA']


def build_trainer(args):
    if args.model_type == 'DRL':
        from models.DRL.Trainer import Trainer
    elif args.model_type == 'MCM':
        from models.MCM.Trainer import Trainer
    elif args.model_type == 'Perceiver':
        from models.Perceiver.Trainer import Trainer
    elif args.model_type in CLASSIC_MODELS:
        from models.Classic.Trainer import Trainer
    else:
        raise ValueError(f"Unknown model type {args.model_type}")

    return Trainer

def load_configs(args):
    dict_to_import = 'model_config_' + args.model_type
    module_name = 'configs'
    module = importlib.import_module(module_name)
    if args.model_type in CLASSIC_MODELS:
        model_config = getattr(module, 'model_config_CLASSIC')
    else:        
        model_config = getattr(module, dict_to_import)

    if args.model_type == 'Perceiver':
        model_config = replace_config(args, model_config)

    return model_config

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

def replace_config(args, model_config):
    model_config['num_heads'] = args.num_heads if args.num_heads is not None else model_config['num_heads']
    model_config['num_layers'] = args.num_layers if args.num_layers is not None else model_config['num_layers']
    model_config['hidden_dim'] = args.hidden_dim if args.hidden_dim is not None else model_config['hidden_dim']
    model_config['mlp_ratio'] = args.mlp_ratio if args.mlp_ratio is not None else model_config['mlp_ratio']
    model_config['dropout_prob'] = args.dropout_prob if args.dropout_prob is not None else model_config['dropout_prob']
    model_config['drop_col_prob'] = args.drop_col_prob if args.drop_col_prob is not None else model_config['drop_col_prob']
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

