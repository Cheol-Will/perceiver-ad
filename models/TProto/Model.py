import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import BaseDecoder, BaseEncoder

class TProto(nn.Module):
    def __init__(
        self, 
        num_features,
        hidden_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout_prob,
        num_prototypes,
        temperature,
        use_flash_attn: bool = False,
        depth_enc: int = None,
        depth_dec: int = None,
    ):
        super().__init__()
        depth_enc = depth if depth_enc is None else depth_enc
        depth_dec = 1 if depth_dec is None else depth_dec

        self.encoder = BaseEncoder(
            num_features, hidden_dim, depth_enc, num_heads,
            mlp_ratio, dropout_prob, use_flash_attn
        )
        self.decoder = BaseDecoder(
            num_features, hidden_dim, depth_dec, num_heads, 
            mlp_ratio, dropout_prob, use_flash_attn
        )
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        self.pos_encoding = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        self.temperature = temperature
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(self, x, return_dict = False):
        z, attn_enc = self.encoder(x)
        z_norm = F.normalize(z, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        weight = F.softmax(z_norm @ p_norm.t() / self.temperature, dim=-1) # (B, N)
        z_hat = weight @ self.prototypes
        x_hat, attn_dec = self.decoder(z_hat, self.pos_encoding)
        if not self.training:
            x = x.float() # turn off amp
            x_hat = x_hat.float()        
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=-1)

        if return_dict:
            return {
                'reconstruction_loss': reconstruction_loss,
                'latent': z,
                'x_hat': x_hat,
                'attn_enc': attn_enc,
                'attn_dec': attn_dec,
            }
        else:
            return reconstruction_loss