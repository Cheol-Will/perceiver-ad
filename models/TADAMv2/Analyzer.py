import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from models.TADAM.Trainer import Trainer 

class Analyzer(Trainer):
    def __init__(self, model_config: dict, train_config: dict):
        super().__init__(model_config, train_config)
        self.model_config = model_config
        self.train_config = train_config

    def get_score_and_latent(self):
        path = os.path.join(self.path, "model.pth")
        path2 = path.replace('results_analysis', 'results') # replace results_analysis with results
        print(path)
        print(path2)
        if os.path.exists(path):
            print(f"Load parameters from {path}")
            self.model.load_state_dict(torch.load(path)) 
            self.model.eval()
        elif os.path.exists(path2):
            print(f"Load parameters from {path2}")
            self.model.load_state_dict(torch.load(path2)) 
            self.model.eval()
        else:
            print("Parameter does not exist. Start training")
            self.training()
            torch.save(self.model.state_dict(), path2)

        model = self.model
        model.eval() 
        print("Build memory bank for evaluation")
        model.build_eval_attn_bank(self.train_loader, self.device, False)

        score, latent = [], []
        x, x_hat  = [], []
        attn_enc, attn_dec = [], []
        knn_score = {f"knn{k}": [] for k in [1, 5, 10, 16, 32, 64]}
        label = []
        with torch.no_grad():
            for (x_input, y_label) in self.train_loader:
                x_input = x_input.to(self.device)
                output = model(x_input)
                
                score.append(output['reconstruction_loss'].data.cpu()) 
                latent.append(output['latent'].data.cpu())
                x.append(x_input.data.cpu())
                x_hat.append(output['x_hat'].data.cpu())
                label.extend(['Train-Normal'] * x_input.size(0))

                # get attn
                batch_attn_enc = torch.stack(output['attn_enc'], dim=1).cpu()
                batch_attn_dec = torch.stack(output['attn_dec'], dim=1).cpu()
                attn_enc.append(batch_attn_enc)
                attn_dec.append(batch_attn_dec)

                # get knn scores (k=1, 5, 10, 16, 32, 64)
                for k, v in output['scores'].items():
                    knn_score[k].append(v)

            for (x_input, y_label) in self.test_loader:
                x_input = x_input.to(self.device)
                output = model(x_input)
                
                score.append(output['reconstruction_loss'].data.cpu()) 
                latent.append(output['latent'].data.cpu())
                x.append(x_input.data.cpu())
                x_hat.append(output['x_hat'].data.cpu())
                y_np = y_label.cpu().numpy()
                batch_labels = np.where(y_np == 0, 'Test-Normal', 'Test-Abnormal')
                label.extend(batch_labels)

                # get attn
                batch_attn_enc = torch.stack(output['attn_enc'], dim=1).cpu()
                batch_attn_dec = torch.stack(output['attn_dec'], dim=1).cpu()
                attn_enc.append(batch_attn_enc)
                attn_dec.append(batch_attn_dec)

                # get knn scores (k=1, 5, 10, 16, 32, 64)
                for k, v in output['scores'].items():
                    knn_score[k].append(v)

        score = torch.cat(score, axis=0).numpy()
        latent = torch.cat(latent, axis=0) 
        x = torch.cat(x, axis=0) 
        x_hat = torch.cat(x_hat, axis=0) 
        attn_enc = torch.cat(attn_enc, axis=0) # (B, L1, H, F+1, F+1)
        attn_dec = torch.cat(attn_dec, axis=0) # (B, L2, H, F+1, F+1)
        knn_score = {k: torch.cat(v, axis=0) for k, v in knn_score.items()}

        return {
            'score': score,
            'label': label,
            'latent': latent,
            'x': x,
            'x_hat': x_hat,
            'attn_enc': attn_enc,
            'attn_dec': attn_dec,
            'knn_score': knn_score,
        }