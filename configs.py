model_config_DRL = {
    'epochs': 200,
    'learning_rate': [0.05, 0.02, 0.01, 0.005],
    'sche_gamma': 0.98,
    'data_dir': 'Data/',
    'batch_size': 512, 
    'en_nlayers': 3,
    'de_nlayers': 3,
    'hidden_dim': 128,
    'random_seed': 42,
    'num_workers': 0,
    'preprocess': 'standard',
    'diversity': True,
    'plearn': False,
    'input_info': True,
    'input_info_ratio': 0.1, # need to tuned?
    'cl': True,
    'cl_ratio': 0.06, # need to tuned?
    'basis_vector_num': 5,
}

model_config_MCM = {
    'epochs': 200,
    'learning_rate': [0.05, 0.02, 0.01, 0.005],
    'sche_gamma': 0.98,
    'mask_num': 15,
    'lambda': [1, 5, 10, 20, 50, 100],
    'data_dir': 'Data/',
    'batch_size': 512, 
    'en_nlayers': 3,
    'de_nlayers': 3,
    'hidden_dim': 256,
    'z_dim': 128,
    'mask_nlayers': 3,
    'random_seed': 42,
    'num_workers': 0,
    'preprocess': 'standard',
}

model_config_Perceiver = {
    'epochs': 200,
    'learning_rate': [0.05, 0.02, 0.01, 0.005],
    'sche_gamma': 0.98,
    'data_dir': 'Data/',
    'batch_size': 512, 
    'num_heads': 4,
    'num_layers': 4,
    'hidden_dim': 64,
    'mlp_ratio': 4.0,
    'dropout_prob': 0.0,
    'drop_col_prob': 0.1, # drop_col_prob should be manually tuned.
    'random_seed': 42,
    'num_workers': 0,
    'preprocess': 'standard',
}


# for classic tad models
model_config_CLASSIC = {
    'data_dir': 'Data/',
    'num_workers': 0,
    'preprocess': 'none',
    'random_seed': 42,
}