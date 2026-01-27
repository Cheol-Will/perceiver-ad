import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.TAEDACLv3.Trainer import Trainer 

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
            obj = torch.load(path, map_location=self.device)
            state = obj if isinstance(obj, dict) else obj.state_dict()
            self.model.load_state_dict(state)
            self.model.eval()
        elif os.path.exists(path2):
            print(f"Load parameters from {path2}")
            obj = torch.load(path2, map_location=self.device)
            state = obj if isinstance(obj, dict) else obj.state_dict()
            self.model.load_state_dict(state)
            self.model.eval()
        else:
            print("Parameter does not exist. Start training")
            self.training()
            torch.save(self.model.state_dict(), path)
        
        model = self.model
        model.eval() 
        print("Build memory bank for evaluation")
        model.build_eval_memory_bank(self.train_loader, self.device, False)

        score, latent = [], []
        x, x_hat, contra_score = [], [], []
        attn_enc, attn_dec = [], []
        label = []
        with torch.no_grad():
            for (x_input, y_label) in self.train_loader:
                x_input = x_input.to(self.device)
                output = model(x_input)
                
                score.append(output['recon_loss'].data.cpu()) 
                contra_score.append(output['contra_score'].data.cpu()) 
                latent.append(output['latent'].data.cpu())
                x.append(x_input.data.cpu())
                x_hat.append(output['x_hat'].data.cpu())
                batch_attn_enc = torch.stack(output['attn_enc'], dim=1).cpu()
                attn_enc.append(batch_attn_enc)
                batch_attn_dec = torch.stack(output['attn_dec'], dim=1).cpu()
                attn_dec.append(batch_attn_dec)
                label.extend(['Train-Normal'] * x_input.size(0))

            for (x_input, y_label) in self.test_loader:
                x_input = x_input.to(self.device)
                output = model(x_input)
                
                score.append(output['recon_loss'].data.cpu()) 
                contra_score.append(output['contra_score'].data.cpu()) 
                latent.append(output['latent'].data.cpu())
                x.append(x_input.data.cpu())
                x_hat.append(output['x_hat'].data.cpu())
                batch_attn_enc = torch.stack(output['attn_enc'], dim=1).cpu()
                attn_enc.append(batch_attn_enc)
                batch_attn_dec = torch.stack(output['attn_dec'], dim=1).cpu()
                attn_dec.append(batch_attn_dec)
                
                y_np = y_label.cpu().numpy()
                batch_labels = np.where(y_np == 0, 'Test-Normal', 'Test-Abnormal')
                label.extend(batch_labels)

        score = torch.cat(score, axis=0).numpy()
        contra_score = torch.cat(contra_score, axis=0).numpy()
        latent = torch.cat(latent, axis=0) 
        x = torch.cat(x, axis=0) 
        x_hat = torch.cat(x_hat, axis=0) 
        attn_enc = torch.cat(attn_enc, axis=0) # (B, L1, H, F+1, F+1)
        attn_dec = torch.cat(attn_dec, axis=0) # (B, L2, H, F+1, F+1)

        return {
            'score': score,
            'contra_score': contra_score,
            'label': label,
            'latent': latent,
            'x': x,
            'x_hat': x_hat,
            'attn_enc': attn_enc,
            'attn_dec': attn_dec,
        }