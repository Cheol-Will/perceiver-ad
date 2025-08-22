import os
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
        return AutoEncoder()
    elif model_type == 'DeepSVDD':
        return DeepSVDD(n_features=model_config['data_dim']) # input dimension should be given
    elif model_type == 'ICL':
        return ICL(hidden_dims='200,400', rep_dim=200) # hyperparamters from ICL paper
    elif model_type == 'NeuTraL':
        return NeuTraL() # 
    elif model_type == 'SLAD':
        return SLAD() # 
    elif model_type == 'GOAD':
        return GOAD() # 
    else: 
        raise ValueError(f"Unknown model type is given: {model_type}")

class Trainer(object):
    def __init__(self, model_config: dict):
        self.device = model_config['device']
        self.model = build_model(model_config)
        self.train_set, self.test_set = get_dataset(model_config)
        self.logger = model_config['logger']
        self.model_config = model_config

    def training(self):
        self.logger.info(self.train_set.data[0]) # to confirm the same data split
        self.logger.info(self.test_set.data[0]) # to confirm the same data split
        print("Training Start.")
        self.model.fit(self.train_set.data)
        print("Training complete.")

    def evaluate(self):
        scores = self.model.decision_function(self.test_set.data)
        test_label = self.test_set.targets
        mse_rauc, mse_ap = aucPerformance(scores, test_label)
        mse_f1 = F1Performance(scores, test_label)
        return mse_rauc, mse_ap, mse_f1