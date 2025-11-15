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
    def __init__(self, model_config: dict, train_config: dict):
        self.train_loader, self.test_loader = get_dataloader(train_config)
        if 'num_latents' not in model_config:
            if train_config['use_num_latents_power_2']:
                model_config['num_latents'] = nearest_power_of_two(int(math.sqrt(model_config['num_features']))) 
            else:
                model_config['num_latents'] = int(math.sqrt(model_config['num_features'])) 

        if train_config['use_num_embeddings_sqrt_NF']:
            # sqrt(NF)
            model_config['num_embeddings'] = int(self.get_num_train()**0.5) * model_config['num_latents'] 
        else:
            # default: sqrt(N)
            model_config['num_embeddings'] = int(self.get_num_train()**0.5)
        if train_config['use_num_embeddings_power_2']:        
            model_config['num_embeddings'] = nearest_power_of_two(model_config['num_embeddings'])

        self.device = train_config['device']
        self.model = PVQVAE(
            **model_config,
        ).to(self.device)

        self.sche_gamma = train_config['sche_gamma']
        self.learning_rate = train_config['learning_rate']
        self.logger = train_config['logger']
        self.epochs = train_config['epochs']
        self.model_config = model_config
        self.train_config = train_config
        

    def get_num_train(self):
        return len(self.train_loader.dataset)

    def training(self):
        print(self.model_config)
        print(self.train_config)

        self.logger.info(self.train_loader.dataset.data[0]) # to confirm the same data split
        self.logger.info(self.test_loader.dataset.data[0]) # to confirm the same data split

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")

        for epoch in range(self.epochs):
            running_loss = 0.0
            running_recon_loss = 0.0
            running_vq_loss = 0.0
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                recon_loss, vq_loss = self.model(x_input) # (B) -> scalar
                loss = recon_loss + vq_loss
                loss = loss.mean()
                running_loss += loss.item()
                running_recon_loss += recon_loss.mean().item()
                running_vq_loss += vq_loss.mean().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t running_recon_loss={:.4f}\t running_vq_loss={:.4f}\t'
            running_loss = running_loss / len(self.train_loader)
            running_recon_loss = running_recon_loss / len(self.train_loader)
            running_vq_loss = running_vq_loss / len(self.train_loader)
            self.logger.info(info.format(epoch, running_loss, running_recon_loss, running_vq_loss))
        print("Training complete.")

    @torch.no_grad()
    def evaluate(self):
        model = self.model
        model.eval()
        score, test_label = [], []
        for step, (x_input, y_label) in enumerate(self.test_loader):
            x_input = x_input.to(self.device)
            recon_loss, vq_loss = model(x_input) # only use recon_loss (?)
            if self.train_config['use_vq_loss_as_score']:
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