#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import numpy as np
import glob
import os
from scipy import io
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

BASELINE_MODELS = ['OCSVM', 'KNN', 'IForest', 'LOF', 'PCA', 'ECOD', 
                   'DeepSVDD', 'GOAD', 'ICL', 'NeuTraL']

npz_files = glob.glob(os.path.join('./Data', '*.npz'))
npz_datanames = [os.path.splitext(os.path.basename(file))[0] for file in npz_files]

mat_files = glob.glob(os.path.join('./Data', '*.mat'))
mat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in mat_files]

dat_files = glob.glob(os.path.join('./Data', '*.data'))
dat_datanames = [os.path.splitext(os.path.basename(file))[0] for file in dat_files]

arff_files = glob.glob(os.path.join('./Data', '*.arff'))
arff_datanames = [os.path.splitext(os.path.basename(file))[0] for file in arff_files]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='Hepatitis')
    parser.add_argument('--model_type', type=str, default='DRL')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--train_ratio', type=float, default=1.0)

    # Model config
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--depth_enc', type=int, default=None)
    parser.add_argument('--depth_dec', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--mlp_ratio', type=float, default=None)
    parser.add_argument('--dropout_prob', type=float, default=None)
    parser.add_argument('--drop_col_prob', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--sim_type', type=str, default=None)
    parser.add_argument('--num_latents', type=int, default=None)
    parser.add_argument('--num_adapters', type=int, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--shrink_thred', type=float, default=None)
    parser.add_argument('--latent_loss_weight', type=float, default=None)
    parser.add_argument('--entropy_loss_weight', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--queue_size', type=int, default=None)
    parser.add_argument('--momentum', type=float, default=None)
    parser.add_argument('--commitment_cost', type=float, default=None)
    parser.add_argument('--reconstruction_weight', type=float, default=None)
    parser.add_argument('--vq_loss_weight', type=float, default=None)
    parser.add_argument('--oe_lambda', type=float, default=None)
    parser.add_argument('--oe_shuffle_ratio', type=float, default=None)
    parser.add_argument('--oe_lambda_memory', type=float, default=None)
    parser.add_argument('--mixup_alpha', type=float, default=None)
    parser.add_argument('--num_prototypes', type=int, default=None)
    parser.add_argument('--sinkhorn_eps', type=float, default=None)
    parser.add_argument('--reconstruction_loss_weight', type=float, default=None)
    parser.add_argument('--contrastive_loss_weight', type=float, default=None)
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--swap_ratio', type=float, default=None)
    parser.add_argument('--num_eval_repeat', type=int, default=None)
    parser.add_argument('--share_mask', action='store_true')

    # Train config
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--sche_gamma', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--min_delta', type=float, default=None)
    parser.add_argument('--contamination_ratio', type=float, default=None)
    parser.add_argument('--latent_ratio', type=float, default=None)
    parser.add_argument('--memory_ratio', type=float, default=None)
    parser.add_argument('--config_file_name', type=str, default=None)

    # Boolean flags (store_true)
    parser.add_argument('--is_weight_sharing', action='store_true')
    parser.add_argument('--use_pos_enc_as_query', action='store_true')
    parser.add_argument('--use_vq_loss_as_score', action='store_true')
    parser.add_argument('--use_entropy_loss_as_score', action='store_true')
    parser.add_argument('--use_mask_token', action='store_true')
    parser.add_argument('--not_use_power_of_two', action='store_true')
    parser.add_argument('--num_memories_not_use_power_of_two', action='store_true')
    parser.add_argument('--num_memories_twice', action='store_true')
    parser.add_argument('--is_recurrent', action='store_true')
    parser.add_argument('--mlp_mixer_encoder', action='store_true')
    parser.add_argument('--mlp_mixer_decoder', action='store_true')
    parser.add_argument('--global_decoder_query', action='store_true')
    parser.add_argument('--mlp_encoder', action='store_true')
    parser.add_argument('--mlp_decoder', action='store_true')
    parser.add_argument('--use_num_latents_power_2', action='store_true')
    parser.add_argument('--use_num_memories_sqrt_NF', action='store_true')
    parser.add_argument('--use_num_memories_power_2', action='store_true')
    parser.add_argument('--use_latent_F', action='store_true')
    parser.add_argument('--not_use_memory', action='store_true')
    parser.add_argument('--not_use_decoder', action='store_true')
    parser.add_argument('--use_pos_embedding', action='store_true')
    parser.add_argument('--use_num_embeddings_sqrt_NF', action='store_true')
    parser.add_argument('--use_num_embeddings_power_2', action='store_true')

    return parser


def _get_store_true_keys(parser):
    """Extract store_true argument names from parser."""
    return {
        action.dest for action in parser._actions
        if isinstance(action, argparse._StoreTrueAction)
    }


def _override_config(config, args_dict, store_true_keys):
    """Override config values with args values."""
    for key in config:
        if key not in args_dict:
            continue
        if key in store_true_keys:
            # store_true: only override if True
            if args_dict[key]:
                config[key] = args_dict[key]
        else:
            # regular args: override if not None
            if args_dict[key] is not None:
                config[key] = args_dict[key]


def load_yaml(args, parser):
    # Determine config file
    if args.config_file_name is not None:
        dict_to_import = args.config_file_name + '.yaml'
        print(f"Load yaml from {args.config_file_name}")
    elif args.model_type in BASELINE_MODELS:
        dict_to_import = 'CLASSIC.yaml'
    else:
        dict_to_import = args.model_type + '.yaml'

    with open(f'configs/{dict_to_import}', 'r') as f:
        configs = yaml.safe_load(f)

    model_config = configs['default']['model_config']
    train_config = configs['default']['train_config']

    # Override with dataset-specific configs
    if args.dataname in configs:
        for k, v in configs[args.dataname].items():
            if k in model_config:
                model_config[k] = v
            if k in train_config:
                train_config[k] = v

    # Override with command-line args
    args_dict = vars(args)
    store_true_keys = _get_store_true_keys(parser)
    
    _override_config(model_config, args_dict, store_true_keys)
    _override_config(train_config, args_dict, store_true_keys)

    # Set required fields
    train_config['model_type'] = args.model_type
    train_config['dataset_name'] = args.dataname
    train_config['base_path'] = args.base_path
    train_config['exp_name'] = args.exp_name
    train_config['runs'] = args.runs
    train_config['train_ratio'] = args.train_ratio
    model_config['num_features'] = get_input_dim(args, train_config)

    return model_config, train_config


def build_trainer(model_config, train_config):
    model_type = train_config['model_type']
    
    if model_type in BASELINE_MODELS:
        from models.Baselines.Trainer import Trainer
    else:
        module = __import__(f'models.{model_type}.Trainer', fromlist=['Trainer'])
        Trainer = module.Trainer
    
    return Trainer(model_config, train_config)

def build_tester(model_config, train_config):
    model_type = train_config['model_type']
    module = __import__(f'models.{model_type}.Tester', fromlist=['Tester'])
    Tester = module.Tester
    return Tester(model_config, train_config)


def get_input_dim(args, config):
    data_dir = config['data_dir']
    
    if args.dataname in npz_datanames:
        path = os.path.join(data_dir, args.dataname + '.npz')
        data = np.load(path)
        return data['X'].shape[-1]
    
    elif args.dataname in mat_datanames:
        path = os.path.join(data_dir, args.dataname + '.mat')
        data = io.loadmat(path)
        return data['X'].shape[-1]
    
    elif args.dataname in dat_datanames:
        path = os.path.join(data_dir, args.dataname + '.data')
        data = pd.read_csv(path, header=None)
        return data['X'].shape[-1]
    
    elif args.dataname in arff_datanames:
        path = os.path.join(data_dir, args.dataname + '.arff')
        data, _ = io.arff.loadarff(path)
        data = pd.DataFrame(data)
        return pd.get_dummies(data.iloc[:, :-1]).shape[-1]
    
    else:
        raise ValueError(f"Unknown dataset {args.dataname}")


def aucPerformance(score, labels):
    roc_auc = roc_auc_score(labels, score)
    ap = average_precision_score(labels, score)
    return roc_auc, ap


def F1Performance(score, target):
    normal_ratio = (target == 0).sum() / len(target)
    score = np.squeeze(score)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = (score > threshold).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(target, pred, average='binary')
    return f1


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("%(message)s")
    
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger