import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from models.TAE.Trainer import Trainer 
from sklearn.manifold import TSNE 


class Analyzer(Trainer):
    def __init__(self, model_config: dict, train_config: dict):
        super().__init__(model_config, train_config)
        self.model_config = model_config
        self.train_config = train_config

    def get_score_and_latent(self):
        parameter_path = os.path.join(self.train_config['base_path'], 'model.pt')
        
        if os.path.exists(parameter_path):
            print("Load parameters")
            self.model.load_state_dict(torch.load(parameter_path)) 
            self.model.eval()
        else:
            print("Parameter does not exist. Start training")
            self.training()
            torch.save(self.model.state_dict(), parameter_path)

        model = self.model
        model.eval() 

        score, label = [], []
        latent, x, x_hat = [], [], []

        for (x_input, y_label) in self.train_loader:
            x_input = x_input.to(self.device)
            output = model(x_input, return_dict=True)

            score.append(output['reconstruction_loss'].data.cpu()) 
            latent.append(output['latent'].data.cpu())
            x.append(x_input.data.cpu())
            x_hat.append(output['x_hat'].data.cpu())
            label.extend(['Train-Normal'] * x_input.size(0))

        for (x_input, y_label) in self.test_loader:
            x_input = x_input.to(self.device)
            output = model(x_input, return_dict=True)
            
            score.append(output['reconstruction_loss'].data.cpu()) 
            latent.append(output['latent'].data.cpu())
            x.append(x_input.data.cpu())
            x_hat.append(output['x_hat'].data.cpu())

            y_np = y_label.cpu().numpy()
            batch_labels = np.where(y_np == 0, 'Test-Normal', 'Test-Abnormal')
            label.extend(batch_labels)

        score = torch.cat(score, axis=0).numpy()
        latent = torch.cat(latent, axis=0) 
        x = torch.cat(x, axis=0) 
        x_hat = torch.cat(x_hat, axis=0) 

        return {
            'score': score,
            'label': label,
            'latent': latent,
            'x': x,
            'x_hat': x_hat,
        }