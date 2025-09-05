import os
import torch
import torch.optim as optim
from DataSet.DataLoader import get_dataloader
from models.PVQVAE.Model import PVQVAE
from utils import aucPerformance, F1Performance
import math

def nearest_power_of_two(x: int) -> int:
    if x < 1:
        return 1
    return 2 ** int(math.floor(math.log2(x)))


class Trainer(object):
    def __init__(self, model_config: dict):
        self.train_loader, self.test_loader = get_dataloader(model_config)
        if 'num_latents' not in model_config:
            if model_config['use_log_num_latents']:
                model_config['num_latents'] = nearest_power_of_two(int(math.log2(model_config['data_dim']))) # sqrt(F)
            else:
                model_config['num_latents'] = nearest_power_of_two(int(math.sqrt(model_config['data_dim']))) # sqrt(F)
        model_config['num_embeddings'] = self.calculate_num_memories() # sqrt(N)
        
        self.device = model_config['device']
        self.sche_gamma = model_config['sche_gamma']
        self.learning_rate = model_config['learning_rate']
        self.model = PVQVAE(
            num_features=model_config['data_dim'],
            num_heads=model_config['num_heads'],
            depth=model_config['depth'],
            hidden_dim=model_config['hidden_dim'],
            mlp_ratio=model_config['mlp_ratio'],
            num_latents=model_config['num_latents'],
            is_weight_sharing=model_config['is_weight_sharing'],
            use_pos_enc_as_query=model_config['use_pos_enc_as_query'],
            beta=model_config['beta'],
            num_embeddings=model_config['num_embeddings'],
        ).to(self.device)
        self.logger = model_config['logger']
        self.model_config = model_config
        self.epochs = model_config['epochs']

    def calculate_num_memories(self):
        n = len(self.train_loader.dataset)
        return nearest_power_of_two(int(math.sqrt(n)))

    def training(self):
        print(self.model_config)
        self.logger.info(self.train_loader.dataset.data[0]) # to confirm the same data split
        self.logger.info(self.test_loader.dataset.data[0]) # to confirm the same data split

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")

        for epoch in range(self.epochs):
            running_loss = 0.0
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                recon_loss, vq_loss = self.model(x_input) # (B) -> scalar
                loss = recon_loss + vq_loss
                loss = loss.mean()
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t'
            running_loss = running_loss / len(self.train_loader)
            self.logger.info(info.format(epoch,loss.cpu()))
        print("Training complete.")

    @torch.no_grad()
    def evaluate(self):
        model = self.model
        model.eval()
        score, test_label = [], []
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)
            recon_loss, vq_loss = model(x_input) # only use recon_loss (?)
            if self.model_config['use_vq_loss_as_score']:
                loss = recon_loss + vq_loss
            else: 
                loss = recon_loss
            # loss = recon_loss
            loss = loss.data.cpu()
            score.append(loss)
            test_label.append(y_label)
        score = torch.cat(score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        rauc, ap = aucPerformance(score, test_label)
        f1 = F1Performance(score, test_label)
        return rauc, ap, f1