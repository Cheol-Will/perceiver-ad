import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from models.layers import BaseEncoder, BaseDecoder

class TAECL(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout_prob,
        temperature,
        contrastive_loss_weight,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.encoder = BaseEncoder(
            num_features, hidden_dim, depth, num_heads,
            mlp_ratio, dropout_prob, use_flash_attn
        )
        self.decoder = BaseDecoder(
            num_features, hidden_dim, num_heads, 
            mlp_ratio, dropout_prob
        )
        self.pos_encoding = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        self.memory_bank = None
        self.temperature = temperature
        self.contrastive_loss_weight = contrastive_loss_weight
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

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
                    z = self.encoder(x_input)
                z = z.float()
            else:
                z = self.encoder(x_input)
            all_keys.append(z.cpu())

        all_keys = torch.cat(all_keys, dim=0).to(device)
        eval_memory = F.normalize(all_keys, dim=-1)
        self.memory_bank = eval_memory
        del all_keys
        torch.cuda.empty_cache()

    def forward(self, x):
        batch_size = x.shape[0]

        z = self.encoder(x)
        x_hat = self.decoder(z, self.pos_encoding)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=-1) # (B,)

        if self.training:
            z_norm = F.normalize(z, dim=-1)
            logits = torch.matmul(z_norm, z_norm.T) / self.temperature # (B, B)
            labels = torch.arange(batch_size).to(x.device)    
            contrastive_loss = F.cross_entropy(logits, labels, reduction='none') 
            contrastive_loss = self.contrastive_loss_weight * contrastive_loss
            loss = reconstruction_loss + contrastive_loss
            output = {
                'loss': loss,
                'reconstruction_loss': reconstruction_loss,
                'contrastive_loss': contrastive_loss,
            }
            return output
        else:
            z = z.float()
            z_norm = F.normalize(z, dim=-1)
            logits = torch.matmul(z_norm, self.memory_bank.T) / self.temperature # (B, N)
            contrastive_score = -torch.logsumexp(logits, dim=-1) * self.contrastive_loss_weight
            output = {
                'reconstruction_loss': reconstruction_loss,
                'contrastive_score': contrastive_score,
                'combined': reconstruction_loss + contrastive_score,
                'latent': z_norm,
                'x_hat': x_hat,
            }
            return output