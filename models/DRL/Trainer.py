import torch
import torch.optim as optim
import torch.nn.functional as F
from DataSet.DataLoader import get_dataloader
from models.DRL.Model import DRL
from utils import aucPerformance, get_logger, F1Performance
import ipdb
import numpy as np
import os

class Trainer(object):
    def __init__(self, model_config: dict, base_path: str):
        self.device = model_config['device']
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sche_gamma = model_config['sche_gamma']
        self.learning_rate = model_config['learning_rate']
        self.model = DRL(model_config).to(self.device)
        self.train_loader, self.test_loader = get_dataloader(model_config)
        self.model_config = model_config
        self.run = model_config['run']
        self.base_path = os.path.join(base_path, str(self.run))
        os.makedirs(self.base_path, exist_ok=True)

    def get_num_instances(self):
        num_train_samples = len(self.train_loader.dataset)
        num_test_samples = len(self.test_loader.dataset)

        return num_train_samples, num_test_samples

    def training(self, epochs):
        train_logger = get_logger(os.path.join(self.base_path, "train_log.log"))
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
                # decomposition loss
                loss = self.model(x_input).mean()

                # alignment loss
                if self.model_config['input_info'] == True:
                    h = self.model.encoder(x_input)
                    x_tilde = self.model.decoder(h)
                    # s_loss = (1-F.cosine_similarity(x_tilde, x_input, dim=-1)).mean() 
                    s_loss = F.cosine_similarity(x_tilde, x_input, dim=-1).mean() * (-1)
                    loss += self.model_config['input_info_ratio'] * s_loss

                # separation loss
                if self.model_config['cl'] == True:
                    h_ = F.softmax(self.model.phi(x_input), dim=1)
                    selected_rows = np.random.choice(h_.shape[0], int(h_.shape[0] * 0.8), replace=False)
                    h_ = h_[selected_rows] # During training, apply dropout-like separation loss.

                    matrix = h_ @ h_.T
                    mol = torch.sqrt(torch.sum(h_**2, dim=-1, keepdim=True)) @ torch.sqrt(torch.sum(h_.T**2, dim=0, keepdim=True))
                    matrix = matrix / mol
                    d_loss =  ((1 - torch.eye(h_.shape[0]).cuda()) * matrix).sum() /(h_.shape[0]) / (h_.shape[0]) # remove inner product with itself.
                    loss += self.model_config['cl_ratio'] * d_loss
                
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t'
            running_loss = running_loss / len(self.train_loader)
            train_logger.info(info.format(epoch,running_loss))
        print("Training complete.")
        train_logger.handlers.clear()

    def evaluate(self):
        model = self.model
        model.eval()
        mse_score, test_label = [], []
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)

            h = model.encoder(x_input)

            weight = F.softmax(self.model.phi(x_input), dim=1)
            h_ = weight@model.basis_vector

            mse = F.mse_loss(h, h_, reduction='none')
            mse_batch = mse.mean(dim=-1, keepdim=True)
            mse_batch = mse_batch.data.cpu()
            mse_score.append(mse_batch)
            test_label.append(y_label)
        mse_score = torch.cat(mse_score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        mse_rauc, mse_ap = aucPerformance(mse_score, test_label)
        mse_f1 = F1Performance(mse_score, test_label)
        return mse_rauc, mse_ap, mse_f1