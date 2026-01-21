import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers import BaseEncoder

class TPair(nn.Module):
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
        contrastive_loss_weight,
        use_flash_attn: bool = False,
        depth_q: int = None,
        depth_k: int = None,
    ):
        super().__init__()
        depth_q = depth if depth_q is None else depth_q
        depth_k = depth if depth_k is None else depth_k
        self.encoder_q = BaseEncoder(
            num_features, hidden_dim, depth_q, num_heads,
            mlp_ratio, dropout_prob, use_flash_attn
        )
        self.encoder_k = BaseEncoder(
            num_features, hidden_dim, depth_k, num_heads,
            mlp_ratio, dropout_prob, use_flash_attn
        )
        self.memory_bank = None
        self.temperature = temperature
        self.mixup_alpha = mixup_alpha
        self.contrastive_loss_weight = contrastive_loss_weight
        
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x, return_dict = False):
        if self.training:
            # mixup during training
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            if lam < 0.5:
                lam = 1 - lam
            index = torch.randperm(x.shape[0]).to(x.device)
            x_aug = lam * x + (1 - lam) * x[index]
        else:
            # no augmentation during inference
            x_aug = x 

        z_q, attn_enc_q = self.encoder_q(x)
        z_k, attn_enc_k = self.encoder_k(x_aug)
       
        # aglinment loss
        alignment_loss = F.mse_loss(z_q, z_k, reduction='none').mean(dim=-1)

        # contrastive loss
        contrastive_loss = F.mse_loss(z_q, z_k, reduction='none').mean(dim=-1)
        z_k_norm = F.normalize(z_k, dim=1)
        logits = torch.matmul(z_k_norm, z_k_norm.T) / self.temperature
        labels = torch.arange(x.shape[0]).to(x.device)
        contrastive_loss = F.cross_entropy(logits, labels, reduction='none')
        contrastive_loss = self.contrastive_loss_weight * contrastive_loss
        loss = alignment_loss + contrastive_loss
        if return_dict:
            return {
                'loss': loss,
                'alignment_loss': alignment_loss,
                'contrastive_loss': contrastive_loss,
                'latent_q': z_q,
                'latent_k': z_k,
                'attn_enc_q': attn_enc_q,
                'attn_enc_k': attn_enc_k,
            }
        else:
            return 