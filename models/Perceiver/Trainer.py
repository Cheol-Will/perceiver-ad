import os
import torch
import torch.optim as optim
from DataSet.DataLoader import get_dataloader
from models.Perceiver.Model import Perceiver
from models.MCM.Loss import LossFunction
from models.MCM.Score import ScoreFunction
from utils import aucPerformance, get_logger, F1Performance


class Trainer(object):
    def __init__(self, model_config: dict, base_path: str):
        self.device = model_config['device']
        self.sche_gamma = model_config['sche_gamma']
        self.learning_rate = model_config['learning_rate']
        self.model = Perceiver(
            num_features=model_config['data_dim'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            hidden_dim=model_config['hidden_dim'],
            mlp_ratio=model_config['mlp_ratio'],
            dropout_prob=model_config['dropout_prob'],
            drop_col_prob=model_config['drop_col_prob'],
        ).to(self.device)
        self.train_loader, self.test_loader = get_dataloader(model_config)
        self.model_config = model_config
        self.seed = model_config['seed']
        self.base_path = os.path.join(base_path, str(self.seed))
        os.makedirs(self.base_path, exist_ok=True)


    def get_num_instances(self):
        num_train_samples = len(self.train_loader.dataset)
        num_test_samples = len(self.test_loader.dataset)

        return num_train_samples, num_test_samples

    def training(self, epochs):
        train_logger = get_logger(os.path.join(self.base_path, 'trian_log.log'))
        num_train_samples = len(self.train_loader.dataset)
        train_logger.info(f"Number of training samples: {num_train_samples}")
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        for epoch in range(epochs):
            running_loss = 0.0
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                loss = self.model(x_input).mean() # (B) -> scalar

                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t'
            running_loss = running_loss / len(self.train_loader)
            train_logger.info(info.format(epoch,loss.cpu()))
        print("Training complete.")
        train_logger.handlers.clear()

    def evaluate(self):
        model = self.model
        model.eval()
        score, test_label = [], []
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)
            loss = self.model(x_input)
            loss = loss.data.cpu()
            score.append(loss)
            test_label.append(y_label)
        score = torch.cat(score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        rauc, ap = aucPerformance(score, test_label)
        f1 = F1Performance(score, test_label)
        return rauc, ap, f1