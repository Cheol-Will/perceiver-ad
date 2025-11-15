import os
import torch
import torch.optim as optim
from DataSet.DataLoader import get_dataloader
from models.MemPAE.Model import MemPAE
from utils import aucPerformance, F1Performance
import math

def nearest_power_of_two(x: int) -> int:
    if x < 1:
        return 1
    return 2 ** int(math.floor(math.log2(x)))


class Trainer(object):
    def __init__(self, model_config: dict, train_config: dict):
        self.train_loader, self.test_loader = get_dataloader(train_config)
        if model_config['num_latents'] is None:
            model_config['num_latents'] = int(math.sqrt(model_config['num_features']))

            if train_config['use_num_latents_power_2']:
                model_config['num_latents'] = nearest_power_of_two(model_config['num_latents'])

            if train_config['latent_ratio'] is not None:
                model_config['num_latents'] *= train_config['latent_ratio']
                model_config['num_latents'] = int(model_config['num_latents'])

        if model_config['num_memories'] is None:
            num_train = self.get_num_train()

            if train_config['use_num_memories_sqrt_NF']:
                # M = sqrt(NF)
                model_config['num_memories'] = int(math.sqrt(num_train * model_config['num_features']))
            else:
                # default: M = sqrt(N) 
                model_config['num_memories'] = int(math.sqrt(num_train))

            if train_config['use_num_memories_power_2']:
                # use power of 2
                model_config['num_memories'] = nearest_power_of_two(model_config['num_memories'])        

            if train_config['memory_ratio'] is not None:
                model_config['num_memories'] *= train_config['memory_ratio']
                model_config['num_memories'] = int(model_config['num_memories'])


        if train_config['use_latent_F']:
            print("Set num_latents = num_features")
            model_config['num_latents'] = model_config['num_features'] 


        self.device = train_config['device']
        self.model = MemPAE(
            **model_config
        ).to(self.device)

        self.sche_gamma = train_config['sche_gamma']
        self.learning_rate = train_config['learning_rate']
        self.logger = train_config['logger']
        self.epochs = train_config['epochs']
        self.model_config = model_config
        self.train_config = train_config
        
    def get_num_train(self):
        num_train = len(self.train_loader.dataset)
        return num_train

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