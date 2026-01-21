import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import BaseDecoder, MLMEncoder

class TMLM(nn.Module):
    def __init__(
        self, 
        num_features,
        hidden_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout_prob,
        use_flash_attn: bool = False,
        mask_ratio: float = 0.2,
        num_eval_repeat: int = 10,
        eval_chunk_size: int = 10, 
    ):
        super().__init__()
        self.encoder = MLMEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_prob=dropout_prob,
            use_flash_attn=use_flash_attn
        )
        self.decoder = BaseDecoder(
            num_features=num_features, 
            hidden_dim=hidden_dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            dropout_prob=dropout_prob
        )
        self.pos_encoding = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        
        self.mask_ratio = mask_ratio
        self.num_eval_repeat = int(num_eval_repeat)
        self.eval_chunk_size = eval_chunk_size

        len_keep = int(num_features * (1 - mask_ratio))
        
        # (R, F)
        noise = torch.rand(self.num_eval_repeat, num_features)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        
        eval_masks = torch.zeros(self.num_eval_repeat, num_features)
        eval_masks.scatter_(1, ids_keep, 1.0)
        
        self.register_buffer('eval_masks', eval_masks) # (R, F)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def _generate_mask(self, x):
        batch_size, num_features = x.shape
        len_keep = int(num_features * (1 - self.mask_ratio))
        
        noise = torch.rand(batch_size, num_features, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        
        mask = torch.zeros_like(x)
        mask.scatter_(1, ids_keep, 1.0)
        return mask

    def forward(self, x, return_dict = False):
        if self.training:
            mask = self._generate_mask(x)
            z, attn_enc = self.encoder(x, mask)
            x_hat, attn_dec = self.decoder(z, self.pos_encoding)
            
            reconstruction_loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=-1)
            
            if return_dict:
                return {
                    'reconstruction_loss': reconstruction_loss,
                    'latent': z,
                    'x_hat': x_hat
                }
            return reconstruction_loss
            
        else:
            batch_size, num_features = x.shape
            R = self.num_eval_repeat
            chunk_size = self.eval_chunk_size
            
            loss_list = []
            x_hat_list = []
            latent_list = []
            
            for start_idx in range(0, R, chunk_size):
                end_idx = min(start_idx + chunk_size, R)
                
                mask_chunk = self.eval_masks[start_idx:end_idx] # (r, F)
                current_r = mask_chunk.shape[0]
                
                x_expanded = x.unsqueeze(1).repeat(1, current_r, 1).view(batch_size * current_r, num_features)
                mask_expanded = mask_chunk.unsqueeze(0).expand(batch_size, -1, -1).reshape(batch_size * current_r, num_features)
                
                z_chunk, attn_enc = self.encoder(x_expanded, mask_expanded) # (B*r, D)
                x_hat_chunk, attn_dec = self.decoder(z_chunk, self.pos_encoding) # (B*r, F)
                
                x_expanded_float = x_expanded.float()
                x_hat_chunk_float = x_hat_chunk.float()
                
                loss_chunk = F.mse_loss(x_hat_chunk_float, x_expanded_float, reduction='none').mean(dim=-1) # (B*r)
                
                loss_list.append(loss_chunk.view(batch_size, current_r))
                x_hat_list.append(x_hat_chunk.view(batch_size, current_r, num_features))
                
                if return_dict:
                    latent_list.append(z_chunk.view(batch_size, current_r, -1))
            
            loss_all = torch.cat(loss_list, dim=1) 
            x_hat_all = torch.cat(x_hat_list, dim=1)
            
            reconstruction_loss = loss_all.mean(dim=1) # (B)
            x_hat = x_hat_all.mean(dim=1) # (B, F)
            
            if not self.training:
                x = x.float() 
                x_hat = x_hat.float()

            if return_dict:
                latent_all = torch.cat(latent_list, dim=1).mean(dim=1)
                return {
                    'reconstruction_loss': reconstruction_loss,
                    'latent': latent_all, 
                    'x_hat': x_hat,    
                }
            else:
                return reconstruction_loss