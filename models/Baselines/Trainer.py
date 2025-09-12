import os
import numpy as np
import torch
from utils import aucPerformance, F1Performance
from DataSet.DataLoader import get_dataset

def build_model(model_config):
    """
    Use default hyperparameters for all models except ICL.
    For ICL, use reported default hyperparameters from ICL paper.


    Since default setting in DataLoader(drop_last=True) of pyod package,
    when train size is smaller than batch size. 
    NO Error returned. 
    """
    model_type = model_config['model_type']
    batch_size = model_config['batch_size']
    if model_type == 'KNN':
        from pyod.models.knn import KNN
        return KNN()
    elif model_type == 'OCSVM':
        from pyod.models.ocsvm import OCSVM
        return OCSVM()
    elif model_type == 'IForest':
        from pyod.models.iforest import IForest
        return IForest()
    elif model_type == 'LOF':
        from pyod.models.lof import LOF
        return LOF()
    elif model_type == 'PCA':
        from pyod.models.pca import PCA
        return PCA()
    elif model_type == 'ECOD':
        from pyod.models.ecod import ECOD
        return ECOD()
    # elif model_type == 'AutoEncoder':
    #     from pyod.models.auto_encoder import AutoEncoder        
    #     return AutoEncoder(batch_size=batch_size)
        # return AutoEncoder(batch_size=batch_size, hidden_neuron_list=[64, 64, 64], )
    elif model_type == 'DeepSVDD':
        from pyod.models.deep_svdd import DeepSVDD
        # return DeepSVDD(batch_size=512, n_features=model_config['data_dim']) # input dimension should be given
        return DeepSVDD(batch_size=batch_size, n_features=model_config['num_features'])
    elif model_type == 'ICL':
        from deepod.models.tabular.icl import ICL
        return ICL(batch_size=batch_size)
    elif model_type == 'NeuTraL':
        from deepod.models.tabular.neutral import NeuTraL
        return NeuTraL(batch_size=batch_size) 
    elif model_type == 'SLAD':
        from deepod.models.tabular.slad import SLAD
        return SLAD(batch_size=batch_size)
    elif model_type == 'GOAD':
        from deepod.models.tabular.goad import GOAD
        return GOAD(batch_size=batch_size)
    else: 
        raise ValueError(f"Unknown model type is given: {model_type}")

class Trainer(object):
    def __init__(self, model_config: dict, train_config: dict):
        # get dataset and then build model
        # since some model changes random seed.
        model_config['model_type'] = train_config['model_type']
        self.train_set, self.test_set = get_dataset(train_config) #
        self.model = build_model(train_config)
        self.logger = train_config['logger']
        self.model_config = model_config
        self.train_config = train_config

    @staticmethod
    def _to_numpy(x, dtype=np.float32):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(dtype, copy=False)
        return np.asarray(x, dtype=dtype)

    def training(self):
        self.logger.info(self.train_set.data[0]) # to confirm the same data split
        self.logger.info(self.test_set.data[0]) # to confirm the same data split
        print("Training Start.")

        X_train = self._to_numpy(self.train_set.data) 
        self.model.fit(X_train)
        print("Training complete.")

    def evaluate(self):
        X_test = self._to_numpy(self.test_set.data)
        scores = self.model.decision_function(X_test)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12) # some abnormal has large input; thus output high anomaly score
        scores_arr = np.asarray(scores, dtype=np.float64).ravel()
        y_test = self._to_numpy(self.test_set.targets, dtype=np.int64).ravel()
        mse_rauc, mse_ap = aucPerformance(scores_arr, y_test)
        mse_f1 = F1Performance(scores_arr, y_test)
        return mse_rauc, mse_ap, mse_f1