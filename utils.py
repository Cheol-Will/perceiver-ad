#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

def build_trainer(args):
    if args.model_type == 'DRL':
        from models.DRL.Trainer import Trainer
    elif args.model_type == 'MCM':
        from models.MCM.Trainer import Trainer
    elif args.model_type == 'Perceiver':
        from models.Perceiver.Trainer import Trainer
    else:
        raise ValueError(f"Unknown model type {args.model_type}")

    return Trainer

def replace_config(args, model_config):
    # for experiment
    model_config['num_heads'] = args.num_heads
    model_config['num_layers'] = args.num_layers
    model_config['hidden_dim'] = args.hidden_dim
    model_config['mlp_ratio'] = args.mlp_ratio
    model_config['dropout_prob'] = args.dropout_prob
    model_config['drop_col_prob'] = args.drop_col_prob
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

