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
import importlib
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

BASELINE_MODELS = ['OCSVM', 'KNN', 'IForest', 'LOF', 'PCA', 'ECOD', 
                   'DeepSVDD', 'GOAD', 'ICL', 'NeuTraL', ] # 'AutoEncoder'

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

    # Experiment 
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--sche_gamma', type=float, default=None)

    # Experiment
    parser.add_argument('--mlp_ratio', type=float, default=None)
    parser.add_argument('--dropout_prob', type=float, default=None)
    parser.add_argument('--drop_col_prob', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--sim_type', type=str, default=None)
    parser.add_argument('--num_repeat', type=int, default=None)
    parser.add_argument('--is_weight_sharing', action='store_true')
    parser.add_argument('--use_pos_enc_as_query', action='store_true')
    parser.add_argument('--num_latents', type=int, default=None)
    parser.add_argument('--num_adapters', type=int, default=None)
    parser.add_argument('--use_vq_loss_as_score', action='store_true')    
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--shrink_thred', type=float, default=None)
    parser.add_argument('--latent_loss_weight', type=float, default=None)
    parser.add_argument('--entropy_loss_weight', type=float, default=None)
    parser.add_argument('--use_entropy_loss_as_score', action='store_true')    
    parser.add_argument('--use_mask_token', action='store_true')    
    parser.add_argument('--not_use_power_of_two', action='store_true')    
    parser.add_argument('--num_memories_not_use_power_of_two', action='store_true')    
    parser.add_argument('--num_memories_twice', action='store_true')    
    parser.add_argument('--is_recurrent', action='store_true')    
    parser.add_argument('--contamination_ratio', type=float, default=None)    
    parser.add_argument('--latent_ratio', type=float, default=None)    
    parser.add_argument('--memory_ratio', type=float, default=None)    
    parser.add_argument('--top_k', type=int, default=None)    
    parser.add_argument('--mlp_mixer_encoder', action='store_true')    
    parser.add_argument('--mlp_mixer_decoder', action='store_true')    
    parser.add_argument('--global_decoder_query', action='store_true')    
    parser.add_argument('--mlp_encoder', action='store_true')    
    parser.add_argument('--mlp_decoder', action='store_true')    
    parser.add_argument('--use_num_latents_power_2', action='store_true')
    parser.add_argument('--use_num_memories_sqrt_NF', action='store_true')
    parser.add_argument('--use_num_memories_power_2', action='store_true')
    parser.add_argument('--use_latent_F', action='store_true')
    parser.add_argument('--config_file_name', type=str, default=None)
    parser.add_argument('--not_use_memory', action='store_true')
    parser.add_argument('--not_use_decoder', action='store_true')
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--min_delta', type=float, default=1e-5)
    parser.add_argument('--use_pos_embedding', action='store_true')

    # Outlier Exposure
    parser.add_argument('--oe_lambda', type=float, default=1.0)
    parser.add_argument('--oe_shuffle_ratio', type=float, default=0.3)
    parser.add_argument('--oe_lambda_memory', type=float, default=0.0)

    # VQVAE
    parser.add_argument('--vq_loss_weight', type=float, default=None)
    parser.add_argument('--use_num_embeddings_sqrt_NF', action='store_true')
    parser.add_argument('--use_num_embeddings_power_2', action='store_true')
    
    return parser


def load_yaml(args):    
    if args.config_file_name is not None:
        dict_to_import = args.config_file_name + '.yaml'
        print(f"Load yaml from {args.config_file_name}")
    else:
        dict_to_import = args.model_type + '.yaml'

    if args.model_type in BASELINE_MODELS:
        dict_to_import = 'CLASSIC.yaml'

    with open(f'configs/{dict_to_import}', 'r') as f:
        configs = yaml.safe_load(f)

    model_config = configs['default']['model_config']
    train_config = configs['default']['train_config']

    if args.model_type in ['OELATTE', 'Perceiver', 'RIN', 'MCMPAE', 'PAE', 'PAEKNN', 'PVAE', 'PVQVAE', 'MemPAE', 'TripletMemPAE', 'PairMemPAE']:
        model_config = replace_transformer_config(args, model_config)
    elif args.model_type in ['MemAE', 'MultiMemAE', 'RINMLP']:
        model_config = replace_mlp_config(args, model_config)
    
    if args.model_type in ['PVQVAE']:
        model_config['vq_loss_weight'] = args.vq_loss_weight 
        train_config['use_vq_loss_as_score'] = args.use_vq_loss_as_score 
        train_config['use_num_embeddings_sqrt_NF'] = args.use_num_embeddings_sqrt_NF
        train_config['use_num_embeddings_power_2'] = args.use_num_embeddings_power_2
        train_config['use_num_latents_power_2'] = args.use_num_latents_power_2

    if args.model_type in ['MemPAE', 'MemSet']:
        train_config['use_num_memories_sqrt_NF'] = args.use_num_memories_sqrt_NF
        train_config['use_num_memories_power_2'] = args.use_num_memories_power_2
        train_config['use_num_latents_power_2'] = args.use_num_latents_power_2
        train_config['use_latent_F'] = args.use_latent_F
        train_config['patience'] = args.patience
        train_config['min_delta'] = args.min_delta
        model_config['not_use_memory'] = args.not_use_memory
        model_config['not_use_decoder'] = args.not_use_decoder
    if args.model_type in ['MemSet']:
        model_config['use_pos_embedding'] = args.use_pos_embedding

    if args.model_type in ['PAE']: 
        train_config['use_latent_F'] = args.use_latent_F

    # Replace hyperparameters with data specific ones. 
    if args.dataname in configs:
        for k, v in configs[args.dataname].items():
            if k in model_config:
                model_config[k] = v
            if k in train_config:
                train_config[k] = v
            if k in ['patience', 'min_delta']:
                train_config[k] = v

    train_config['model_type'] = args.model_type
    train_config['dataset_name'] = args.dataname
    train_config['train_ratio'] = args.train_ratio
    train_config['base_path'] = args.base_path    
    train_config['learning_rate'] = args.learning_rate if args.learning_rate is not None else train_config['learning_rate']
    train_config['sche_gamma'] = args.sche_gamma if args.sche_gamma is not None else train_config['sche_gamma']
    train_config['not_use_power_of_two'] = args.not_use_power_of_two # default False -> use power of two
    train_config['num_memories_not_use_power_of_two'] = args.num_memories_not_use_power_of_two # defualt None
    train_config['num_memories_twice'] = args.num_memories_twice # defualt None
    train_config['contamination_ratio'] = args.contamination_ratio # defualt None
    train_config['latent_ratio'] = args.latent_ratio # defualt None
    train_config['memory_ratio'] = args.memory_ratio # defualt None
    model_config['num_features'] = get_input_dim(args, train_config)

    return model_config, train_config


def build_trainer(model_config, train_config):
    model_type = train_config['model_type']
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
    elif model_type == 'TripletMemPAE':
        from models.TripletMemPAE.Trainer import Trainer        
    elif model_type == 'PairMemPAE':
        from models.PairMemPAE.Trainer import Trainer        
    elif model_type == 'MultiMemAE':
        from models.MultiMemAE.Trainer import Trainer
    elif model_type == 'RINMLP':
        from models.RINMLP.Trainer import Trainer        
    elif model_type == 'PAE':
        from models.PAE.Trainer import Trainer
    elif model_type == 'PAEKNN':
        from models.PAEKNN.Trainer import Trainer                
    elif model_type == 'PVAE':
        from models.PVAE.Trainer import Trainer
    elif model_type == 'PVQVAE':
        from models.PVQVAE.Trainer import Trainer                
    elif model_type == 'AutoEncoder':
        from models.AutoEncoder.Trainer import Trainer        
    elif model_type == 'PDRL':
        from models.PDRL.Trainer import Trainer
    elif model_type == 'MCMPAE':
        from models.MCMPAE.Trainer import Trainer        
    elif model_type == 'NPTAD':
        from models.NPTAD.Trainer import Trainer        
    elif model_type == 'RetAug':
        from models.RetAug.Trainer import Trainer        
    elif model_type == 'MemSet':
        from models.MemSet.Trainer import Trainer        
    elif model_type == 'OELATTE':
        from models.OELATTE.Trainer import Trainer        
    elif model_type in BASELINE_MODELS:
        from models.Baselines.Trainer import Trainer
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return Trainer(model_config, train_config)

def get_input_dim(args, model_config):
    if args.dataname in npz_datanames:
        path = os.path.join(model_config['data_dir'], args.dataname + '.npz')
        data = np.load(path)
    elif args.dataname in mat_datanames:
        path = os.path.join(model_config['data_dir'], args.dataname + '.mat')
        data = io.loadmat(path)
    elif args.dataname in dat_datanames:
        path = os.path.join(model_config['data_dir'], args.dataname + '.data')  
        data = pd.read_csv(path, header=None) 
        print(data)
        # data = io.loadmat(path)
    elif args.dataname in arff_datanames:
        path = os.path.join(model_config['data_dir'], args.dataname + '.arff')
        data, _ = io.arff.loadarff(path)
        data = pd.DataFrame(data)
        samples = pd.get_dummies(data.iloc[:, :-1]).to_numpy()
        labels = data.iloc[:, -1].values
    else:
        raise ValueError(f"Unknown dataset {args.dataname}")
    if args.dataname not in arff_datanames:
        dim = data['X'].shape[-1]

    return dim

def replace_transformer_config(args, model_config):
    model_config['num_heads'] = args.num_heads if args.num_heads is not None else model_config['num_heads']
    model_config['depth'] = args.depth if args.depth is not None else model_config['depth']
    model_config['hidden_dim'] = args.hidden_dim if args.hidden_dim is not None else model_config['hidden_dim']
    model_config['mlp_ratio'] = args.mlp_ratio if args.mlp_ratio is not None else model_config['mlp_ratio']
    model_config['dropout_prob'] = args.dropout_prob if args.dropout_prob is not None else model_config['dropout_prob']

    if args.model_type in ['Perceiver', 'RIN']:
        model_config['drop_col_prob'] = args.drop_col_prob if args.drop_col_prob is not None else model_config['drop_col_prob']
    
    if args.model_type == 'OELATTE':
        model_config['oe_lambda'] = args.oe_lambda # 
        model_config['oe_shuffle_ratio'] = args.oe_shuffle_ratio # 
        model_config['oe_lambda_memory'] = args.oe_lambda_memory # 

    if args.model_type in ['OELATTE', 'MCMPAE', 'PAE', 'MemPAE', 'TripletMemPAE', 'PairMemPAE', 'PAEKNN', 'PVAE', 'PVQVAE', 'PDRL', 'MemSet']:
        model_config['is_weight_sharing'] = args.is_weight_sharing # default False
        model_config['use_pos_enc_as_query'] = args.use_pos_enc_as_query # default False
        model_config['use_mask_token'] = args.use_mask_token # default False
            
        if args.num_latents is not None:
            model_config['num_latents'] = args.num_latents


        if args.model_type in['MemPAE', 'PAE', 'MemSet']: 
            model_config['global_decoder_query'] = args.global_decoder_query # None
            model_config['mlp_mixer_decoder'] = args.mlp_mixer_decoder # None
            model_config['mlp_mixer_encoder'] = args.mlp_mixer_encoder # None
            model_config['mlp_encoder'] = args.mlp_encoder # None
            model_config['mlp_decoder'] = args.mlp_decoder # None
            

        if args.model_type in ['MemPAE', 'TripletMemPAE', 'PairMemPAE', 'MemSet']: 
            model_config['sim_type'] = args.sim_type if args.sim_type is not None else model_config['sim_type']
            model_config['temperature'] = args.temperature if args.temperature is not None else model_config['temperature']
            model_config['shrink_thred'] = args.shrink_thred if args.shrink_thred is not None else model_config['shrink_thred']
            model_config['latent_loss_weight'] = args.latent_loss_weight # defualt None
            model_config['entropy_loss_weight'] = args.entropy_loss_weight # defualt None
            model_config['use_entropy_loss_as_score'] = args.use_entropy_loss_as_score # default False
            model_config['is_recurrent'] = args.is_recurrent # default False
            model_config['top_k'] = args.top_k # None

        if args.model_type in ['PVAE', 'PVQVAE']:
            model_config['beta'] = args.beta if args.beta is not None else model_config['beta'] 

    return model_config

def replace_mlp_config(args, model_config):
    # model_config['depth'] = args.depth if args.depth is not None else model_config['depth']
    # model_config['depth'] = args.depth if args.depth is not None else model_config['depth']
    model_config['hidden_dim'] = args.hidden_dim if args.hidden_dim is not None else model_config['hidden_dim']

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

def load_yaml_multitask(args):
    dict_to_import = args.model_type + '.yaml'
    with open(f'configs/{dict_to_import}', 'r') as f:
        configs = yaml.safe_load(f)

    model_config = configs['default']['model_config']
    train_config = configs['default']['train_config']
    train_config['model_type'] = args.model_type
    train_config['train_ratio'] = args.train_ratio
    train_config['base_path'] = args.base_path
    
    # Override model_config with command-line arguments if provided
    if args.num_heads is not None:
        model_config['num_heads'] = args.num_heads
    if args.depth is not None:
        model_config['depth'] = args.depth
    if args.hidden_dim is not None:
        model_config['hidden_dim'] = args.hidden_dim
    if args.num_latents is not None:
        model_config['num_latents'] = args.num_latents
    if args.num_memories is not None:
        model_config['num_memories'] = args.num_memories
    if args.is_weight_sharing:
        model_config['is_weight_sharing'] = args.is_weight_sharing
    if args.temperature is not None:
        model_config['temperature'] = args.temperature
    if args.sim_type is not None:
        model_config['sim_type'] = args.sim_type
    
    # Override train_config with command-line arguments if provided
    if args.learning_rate is not None:
        train_config['learning_rate'] = args.learning_rate
    if args.sche_gamma is not None:
        train_config['sche_gamma'] = args.sche_gamma

    return model_config, train_config