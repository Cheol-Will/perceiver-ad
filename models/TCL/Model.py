import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
from models.layers import BaseEncoder

class TCL(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout_prob,
        temperature,
        mixup_alpha,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.encoder = BaseEncoder(
            num_features, hidden_dim, depth, num_heads,
            mlp_ratio, dropout_prob, use_flash_attn
        )
        self.memory_bank = None
        self.temperature = temperature
        self.mixup_alpha = mixup_alpha

    @torch.no_grad()
    def empty_eval_memory_bank(self,):
        self.memory_bank = None

    @torch.no_grad()
    def build_eval_memory_bank(self, train_loader, device, use_amp=False):
        """
        Build memory bank from all training samples for evaluation.
        """ 
        self.eval()
        all_keys = []
        for (x_input, y_label) in train_loader:
            x_input = x_input.to(device)
            if use_amp:
                with autocast():
                    z, attn = self.encoder(x_input)
                z = z.float()
            else:
                z, attn = self.encoder(x_input)
            all_keys.append(z.cpu())

        all_keys = torch.cat(all_keys, dim=0).to(device)
        eval_memory = F.normalize(all_keys, dim=-1)
        self.memory_bank = eval_memory
        del all_keys
        torch.cuda.empty_cache()

    def forward(self, x):
        batch_size = x.shape[0]

        if self.training:
            if self.mixup_alpha > 0:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                if lam < 0.5: 
                    lam = 1 - lam
                index = torch.randperm(batch_size).to(x.device)
                x_aug = lam * x + (1 - lam) * x[index]
            else:
                x_aug = x

            query, _ = self.encoder(x)
            key, _ = self.encoder(x_aug)
            query = F.normalize(query, dim=-1)
            key = F.normalize(key, dim=-1)
            logits = torch.matmul(query, key.T) / self.temperature # (B, B)
            labels = torch.arange(batch_size).to(x.device)
            contrastive_loss = F.cross_entropy(logits, labels, reduction='none') 
            
            output = {
                'loss': contrastive_loss,
            }
            return output
            
        else:
            z, _ = self.encoder(x)
            z = z.float()
            z_norm = F.normalize(z, dim=-1)
            
            logits = torch.matmul(z_norm, self.memory_bank.T) / self.temperature
            contrastive_score = -torch.logsumexp(logits, dim=-1)
            
            output = {
                'contrastive_score': contrastive_score,
                'latent': z_norm,
            }
            return output