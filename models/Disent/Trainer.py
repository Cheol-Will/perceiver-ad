import torch
import torch.optim as optim
import torch.nn.functional as F
from DataSet.DataLoader import get_dataloader
from models.Disent.Model import Disent
from utils import aucPerformance, get_logger, F1Performance
import ipdb
import numpy as np
import os

class Trainer(object):
    def __init__(self, model_config: dict, base_path: str):
        self.train_loader, self.test_loader = get_dataloader(model_config)
        self.device = model_config['device']
        self.sche_gamma = model_config['sche_gamma']
        self.learning_rate = model_config['learning_rate']
        self.model = Disent(model_config).to(self.device)
        # Disent-AD need its own data loader since it has patch spliting.
        self.train_loader, self.test_loader = get_dataloader(model_config)
        self.logger = model_config['logger']
        self.model_config = model_config
        self.epochs = model_config['epochs']

    def training(self, epochs):
        self.logger.info(self.train_loader.dataset.data[0]) # to confirm the same data split
        self.logger.info(self.test_loader.dataset.data[0]) # to confirm the same data split

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()

        train_logger = get_logger(os.path.join(self.base_path, "DRL.log"))
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        for epoch in range(epochs):
            running_loss = 0.0
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                # decomposition loss
                recon_loss, dis_loss = self.model(x_input).mean()
                loss = recon_loss + dis_loss
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t'
            running_loss = running_loss / len(self.train_loader)
            train_logger.info(info.format(epoch,running_loss))

        print("Training complete.")

    @torch.no_grad()
    def evaluate(self):
        model = self.model
        model.eval()
        mse_score, test_label = [], []
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)

            mse = model(x_input)
            mse_batch = mse.mean(dim=-1, keepdim=True)
            mse_batch = mse_batch.data.cpu()
            mse_score.append(mse_batch)
            test_label.append(y_label)
        mse_score = torch.cat(mse_score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        mse_rauc, mse_ap = aucPerformance(mse_score, test_label)
        mse_f1 = F1Performance(mse_score, test_label)
        return mse_rauc, mse_ap, mse_f1