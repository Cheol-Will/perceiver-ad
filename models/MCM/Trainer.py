import os
import torch
import torch.optim as optim
from DataSet.DataLoader import get_dataloader
from models.MCM.Model import MCM
from models.MCM.Loss import LossFunction
from models.MCM.Score import ScoreFunction
from utils import aucPerformance, get_logger, F1Performance

class Trainer(object):
    def __init__(self, model_config: dict, base_path: str):
        self.device = model_config['device']
        self.sche_gamma = model_config['sche_gamma']
        self.learning_rate = model_config['learning_rate']
        self.model = MCM(model_config).to(self.device)
        self.loss_fuc = LossFunction(model_config).to(self.device)
        self.score_func = ScoreFunction(model_config).to(self.device)
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
        min_loss = 100
        for epoch in range(epochs):
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                x_pred, z, masks = self.model(x_input)
                loss, mse, divloss = self.loss_fuc(x_input, x_pred, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t mse={:.4f}\t divloss={:.4f}\t'
            train_logger.info(info.format(epoch,loss.cpu(),mse.cpu(),divloss.cpu()))
            if loss < min_loss:
                torch.save(self.model, os.path.join(self.base_path, 'model.pth'))
                min_loss = loss
        print("Training complete.")
        train_logger.handlers.clear()

    def evaluate(self):
        model = torch.load(os.path.join(self.base_path, 'model.pth'))
        model.eval()
        mse_score, test_label = [], []
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)
            x_pred, z, masks = self.model(x_input)
            mse_batch = self.score_func(x_input, x_pred)
            mse_batch = mse_batch.data.cpu()
            mse_score.append(mse_batch)
            test_label.append(y_label)
        mse_score = torch.cat(mse_score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        mse_rauc, mse_ap = aucPerformance(mse_score, test_label)
        mse_f1 = F1Performance(mse_score, test_label)
        return mse_rauc, mse_ap, mse_f1