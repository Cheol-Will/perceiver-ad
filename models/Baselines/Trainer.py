import os
import numpy as np
import torch
from utils import aucPerformance, F1Performance
from DataSet.DataLoader import get_dataset

from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.ecod import ECOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.deep_svdd import DeepSVDD
from deepod.models.tabular.icl import ICL
from deepod.models.tabular.neutral import NeuTraL
from deepod.models.tabular.slad import SLAD
from deepod.models.tabular.goad import GOAD


def build_model(model_config):
    """
    Use default hyperparameters for all models except ICL.
    For ICL, use reported default hyperparameters from ICL paper.
    """
    model_type = model_config['model_type']
    if model_type == 'KNN':
        return KNN()
    elif model_type == 'OCSVM':
        return OCSVM()
    elif model_type == 'IForest':
        return IForest()
    elif model_type == 'LOF':
        return LOF()
    elif model_type == 'PCA':
        return PCA()
    elif model_type == 'ECOD':
        return ECOD()
    elif model_type == 'AutoEncoder':
        return AutoEncoder(batch_size=512)
    elif model_type == 'DeepSVDD':
        return DeepSVDD(batch_size=512, n_features=model_config['data_dim']) # input dimension should be given
    elif model_type == 'ICL':
        # return ICL(hidden_dims='200,400', rep_dim=200) # hyperparamters from ICL paper
        return ICL(batch_size=512) 
    elif model_type == 'NeuTraL':
        # return NeuTraL(batch_size=512, hidden_dims='24,24,24,24', rep_dim=24, trans_hidden_dims=24, n_trans=11) #  hyperparamters from ICL paper 
        return NeuTraL(batch_size=512) 
    elif model_type == 'SLAD':
        return SLAD(batch_size=512) # 
    elif model_type == 'GOAD':
        return GOAD(batch_size=512) # 
    else: 
        raise ValueError(f"Unknown model type is given: {model_type}")

class Trainer(object):
    def __init__(self, model_config: dict):
        # get dataset and then build model
        # since some model changes random seed.
        self.device = model_config['device']
        self.train_set, self.test_set = get_dataset(model_config) #
        self.model = build_model(model_config)
        self.logger = model_config['logger']
        self.model_config = model_config

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
        # nan_mask = np.isnan(scores_arr) | np.isinf(scores_arr)

        # if nan_mask.any():
        #     bad_idx = np.where(nan_mask)[0]
        #     print(f"[DEBUG] Found {len(bad_idx)} invalid scores at indices: {bad_idx.tolist()}")

        #     for i in bad_idx:
        #         print(f"idx={i}, score={scores_arr[i]}, input={X_test[i]}, label={self.test_set.targets[i]}")

        y_test = self._to_numpy(self.test_set.targets, dtype=np.int64).ravel()
        mse_rauc, mse_ap = aucPerformance(scores_arr, y_test)
        mse_f1 = F1Performance(scores_arr, y_test)
        return mse_rauc, mse_ap, mse_f1