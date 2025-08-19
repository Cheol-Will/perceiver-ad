import os
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from DataSet.DataLoader import get_dataset
from utils import aucPerformance, get_logger, F1Performance

def build_model(model_type):
    if model_type == 'KNN':
        return KNN
    elif model_type == 'OCSVM':
        return OCSVM
    elif model_type == 'IForest':
        return IForest
    elif model_type == 'LOF':
        return LOF
    elif model_type == 'PCA':
        return PCA
    else: 
        raise ValueError(f"Unknown model type is given: {model_type}")

class Trainer(object):
    def __init__(self, model_config: dict, base_path: str):
        self.device = model_config['device']
        Model = build_model(model_config['model_type'])
        self.model = Model()
        self.train_set, self.test_set = get_dataset(model_config)
        self.model_config = model_config
        self.run = model_config['run']
        self.base_path = os.path.join(base_path, str(self.run))
        os.makedirs(self.base_path, exist_ok=True)

    def get_num_instances(self):
        num_train_samples = len(self.train_set)
        num_test_samples = len(self.test_set)

        return num_train_samples, num_test_samples

    def training(self, epochs = None):
        train_logger = get_logger(os.path.join(self.base_path, "train_log.log"))
        num_train_samples = len(self.train_set)
        train_logger.info(f"Number of training samples: {num_train_samples}")
        print("Training Start.")
        self.model.fit(self.train_set.data)

        print("Training complete.")
        train_logger.handlers.clear()

    def evaluate(self):
        scores = self.model.decision_function(self.test_set.data)
        test_label = self.test_set.targets
        mse_rauc, mse_ap = aucPerformance(scores, test_label)
        mse_f1 = F1Performance(scores, test_label)
        return mse_rauc, mse_ap, mse_f1