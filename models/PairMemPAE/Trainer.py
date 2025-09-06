import os
import torch
import torch.optim as optim
from DataSet.DataLoader import get_dataloader
from models.PairMemPAE.Model import PairMemPAE
from utils import aucPerformance, F1Performance
import math

def nearest_power_of_two(x: int) -> int:
    if x < 1:
        return 1
    return 2 ** int(math.floor(math.log2(x)))


class Trainer(object):
    def __init__(self, model_config: dict):
        self.train_loader, self.test_loader = get_dataloader(model_config)
        
        model_config['num_latents'] = nearest_power_of_two(int(math.sqrt(model_config['data_dim']))) # sqrt(F)

        if model_config['use_small_memory'] == True:
            model_config['num_memories'] = self.calculate_num_memories() # sqrt(N)
            model_config['num_memories'] //=  2 # sqrt(N) // 2
            print("use half of memory size")
        else: 
            model_config['num_memories'] = self.calculate_num_memories() # sqrt(N)

        self.device = model_config['device']
        self.sche_gamma = model_config['sche_gamma']
        self.learning_rate = model_config['learning_rate']
        self.model = PairMemPAE(
            num_features=model_config['data_dim'],
            num_heads=model_config['num_heads'],
            depth=model_config['depth'],
            hidden_dim=model_config['hidden_dim'],
            mlp_ratio=model_config['mlp_ratio'],
            num_latents=model_config['num_latents'],
            num_memories=model_config['num_memories'],
            is_weight_sharing=model_config['is_weight_sharing'],
            temperature=model_config['temperature'],
            sim_type=model_config['sim_type'],
            use_pos_enc_as_query=model_config['use_pos_enc_as_query'],
            shrink_thred=model_config['shrink_thred'],
        ).to(self.device)
        self.logger = model_config['logger']
        self.model_config = model_config
        self.epochs = model_config['epochs']

    def calculate_num_memories(self):
        n = len(self.train_loader.dataset)
        return nearest_power_of_two(int(math.sqrt(n)))

    def training(self):
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
                loss = self.model(x_input).mean() # (B) -> scalar

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
            loss = model(x_input)
            loss = loss.data.cpu()
            score.append(loss)
            test_label.append(y_label)
        score = torch.cat(score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        rauc, ap = aucPerformance(score, test_label)
        f1 = F1Performance(score, test_label)
        return rauc, ap, f1