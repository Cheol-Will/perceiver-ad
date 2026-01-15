import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import BaseDecoder, BaseEncoder

class TAE(nn.Module):
    def __init__(
        self, 
        num_features,
        hidden_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout_prob,
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
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def forward(self, x, return_dict = False):
        z = self.encoder(x)
        x_hat = self.decoder(z, self.pos_encoding)
        if not self.training:
            x = x.float() # turn of amp
            x_hat = x_hat.float()        
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=-1)

        if return_dict:
            return {
                'reconstruction_loss': reconstruction_loss,
                'latent': z,    
            }
        else:
            return reconstruction_loss