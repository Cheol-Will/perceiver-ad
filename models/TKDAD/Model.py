import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import BaseEncoder, MLP

class TKDAD(nn.Module):
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
        
        self.encoder1 = BaseEncoder(
            num_features, hidden_dim, depth, num_heads,
            mlp_ratio, dropout_prob, use_flash_attn
        )
        self.encoder2 = BaseEncoder(
            num_features, hidden_dim, depth, num_heads,
            mlp_ratio, dropout_prob, use_flash_attn
        )

    def forward(self, x, return_dict=False):
        z_1, attn_1 = self.encoder1(x)
        z_2, attn_2 = self.encoder1(x)

        # contrastive loss for z_1
        


    def forward(self, x, return_dict=False):
        z_s, attn_s = self.student(x)
        
        # Teacher forward (on Hypersphere)
        with torch.no_grad():
            z_t, attn_t = self.teacher(x)
            z_t = F.normalize(z_t, p=2, dim=-1) 

        reg_loss = F.mse_loss(z_s, z_t, reduction='none').mean(dim=-1)

        return