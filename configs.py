model_config_DRL = {
    'epochs': 200,
    'learning_rate': 0.05,
    'sche_gamma': 0.98,
    'device': 'cuda:0',
    'data_dir': 'Data/',
    'runs': 1,
    'batch_size': 512, 
    'en_nlayers': 3,
    'de_nlayers': 3,
    'hidden_dim': 128,
    'random_seed': 42,
    'num_workers': 0,
    'preprocess': 'standard'
}