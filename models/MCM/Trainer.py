import os
import numpy as np
import torch
import torch.optim as optim
from DataSet.DataLoader import get_dataloader
from models.MCM.Model import MCM
from models.MCM.Loss import LossFunction
from models.MCM.Score import ScoreFunction
from utils import aucPerformance, F1Performance

class Trainer(object):
    def __init__(self, model_config: dict, train_config: dict):
        self.train_loader, self.test_loader = get_dataloader(train_config)
        self.sche_gamma = train_config['sche_gamma']
        self.device = train_config['device']
        self.learning_rate = train_config['learning_rate']
        self.model = MCM(model_config, train_config).to(self.device)
        self.loss_fuc = LossFunction(model_config).to(self.device)
        self.score_func = ScoreFunction(model_config).to(self.device)
        self.logger = train_config['logger']
        self.train_config = train_config
        self.epochs = train_config['epochs']
        self.path = os.path.join(train_config['base_path'], str(train_config['run']), 'model.pth')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def training(self):
        self.logger.info(self.train_loader.dataset.data[0]) # to confirm the same data split
        self.logger.info(self.test_loader.dataset.data[0]) # to confirm the same data split
        print(f"shape of trainset: {self.train_loader.dataset.data.shape}")
        print(f"shape of testset: {self.test_loader.dataset.data.shape}")
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        min_loss = float('inf')
        for epoch in range(self.epochs):
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                x_pred, z, masks = self.model(x_input)
                loss, mse, divloss = self.loss_fuc(x_input, x_pred, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t mse={:.4f}\t divloss={:.4f}\t'
            self.logger.info(info.format(epoch,loss.cpu(),mse.cpu(),divloss.cpu()))
            # train_logger.info(info.format(epoch,loss.cpu(),mse.cpu(),divloss.cpu()))
            if loss < min_loss:
                torch.save(self.model, self.path)
                min_loss = loss
        print("Training complete.")

    def evaluate(self):
        model = torch.load(self.path)
        model.eval()
        mse_score, test_label = [], []
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)
            x_pred, z, masks = model(x_input)
            mse_batch = self.score_func(x_input, x_pred)
            mse_batch = mse_batch.data.cpu()
            mse_score.append(mse_batch)
            test_label.append(y_label)
        mse_score = torch.cat(mse_score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        mse_score = np.nan_to_num(mse_score, nan=0.0, posinf=1e12, neginf=-1e12) # some abnormal has large input; thus output high anomaly score

        mse_rauc, mse_ap = aucPerformance(mse_score, test_label)
        mse_f1 = F1Performance(mse_score, test_label)
        return mse_rauc, mse_ap, mse_f1