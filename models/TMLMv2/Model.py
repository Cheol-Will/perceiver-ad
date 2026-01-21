import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import SelfAttention, CrossAttention, FeatureTokenizer, OutputProjection

class TMLMCross(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim,
        enc_depth,
        dec_depth,
        num_heads,
        mlp_ratio=4.0,
        dropout_prob=0.0,
        mask_ratio=0.2,
        use_flash_attn=False,
        num_eval_repeat=10,
        eval_chunk_size=10
    ):
        super().__init__()
        self.num_features = num_features
        self.mask_ratio = mask_ratio
        self.num_eval_repeat = num_eval_repeat
        self.eval_chunk_size = eval_chunk_size

        self.feature_tokenizer = FeatureTokenizer(num_features, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_features, hidden_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        self.encoder_blocks = nn.ModuleList([
            SelfAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob, use_flash_attn)
            for _ in range(enc_depth)
        ])
        self.enc_norm = nn.LayerNorm(hidden_dim)

        self.decoder_blocks = nn.ModuleList([
            CrossAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob, use_flash_attn)
            for _ in range(dec_depth)
        ])
        self.dec_norm = nn.LayerNorm(hidden_dim)
        
        self.output_projection = OutputProjection(num_features, hidden_dim)

        len_keep = int(num_features * (1 - mask_ratio))
        noise = torch.rand(num_eval_repeat, num_features)
        ids_shuffle = torch.argsort(noise, dim=1)
        eval_ids_keep = ids_shuffle[:, :len_keep]
        eval_ids_mask = ids_shuffle[:, len_keep:]
        
        self.register_buffer('eval_ids_keep', eval_ids_keep)
        self.register_buffer('eval_ids_mask', eval_ids_mask)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.feature_tokenizer.reset_parameters()
        self.output_projection.reset_parameters()
        
    def _random_masking(self, x):
        N, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]

        x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_keep, ids_keep, ids_mask

    def forward_encoder(self, x, ids_keep):
        x_emb = self.feature_tokenizer(x)
        x_emb = x_emb + self.pos_embed
        
        D = x_emb.shape[-1]
        x_visible = torch.gather(x_emb, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        for blk in self.encoder_blocks:
            x_visible, _ = blk(x_visible)
        
        x_visible = self.enc_norm(x_visible)
        return x_visible

    def forward_decoder(self, x_visible, ids_mask):
        B, N_mask = ids_mask.shape
        D = x_visible.shape[-1]

        mask_tokens = self.mask_token.repeat(B, N_mask, 1)
        pos_mask = torch.gather(self.pos_embed.expand(B, -1, -1), 1, ids_mask.unsqueeze(-1).repeat(1, 1, D))
        
        x_query = mask_tokens + pos_mask

        for blk in self.decoder_blocks:
            x_query, _ = blk(x_query, x_visible, x_visible)
            
        x_query = self.dec_norm(x_query)
        
        # Partial Output Projection
        # Gather embedding matrix corresponding to masked indices
        proj_weight = torch.gather(
            self.output_projection.embedding_matrix.expand(B, -1, -1), 
            1, 
            ids_mask.unsqueeze(-1).repeat(1, 1, D)
        )
        x_rec = (x_query * proj_weight).sum(dim=-1) # (B, N_mask)
        
        return x_rec.unsqueeze(-1) # (B, N_mask, 1) for loss calculation

    def forward(self, x, return_dict=False):
        if self.training:
            x_in = x
            x_emb = x_in.unsqueeze(-1).expand(-1, -1, self.pos_embed.shape[-1])
            _, ids_keep, ids_mask = self._random_masking(x_emb)
            
            x_enc = self.forward_encoder(x_in, ids_keep)
            x_rec = self.forward_decoder(x_enc, ids_mask)
            
            target = torch.gather(x_in.unsqueeze(-1), 1, ids_mask.unsqueeze(-1))
            loss = F.mse_loss(x_rec, target)
            
            if return_dict:
                return {'loss': loss}
            return loss

        else:
            B, F_dim = x.shape
            total_samples = B * self.num_eval_repeat
            x_repeated = x.repeat_interleave(self.num_eval_repeat, dim=0)
            
            all_losses = []
            
            for i in range(0, total_samples, self.eval_chunk_size):
                end = min(i + self.eval_chunk_size, total_samples)
                batch_x = x_repeated[i:end]
                current_batch_size = batch_x.shape[0]
                
                indices = torch.arange(current_batch_size, device=x.device) + i
                mask_indices = indices % self.num_eval_repeat
                
                ids_keep = self.eval_ids_keep[mask_indices]
                ids_mask = self.eval_ids_mask[mask_indices]
                
                x_enc = self.forward_encoder(batch_x, ids_keep)
                x_rec = self.forward_decoder(x_enc, ids_mask)
                
                target = torch.gather(batch_x.unsqueeze(-1), 1, ids_mask.unsqueeze(-1))
                loss_chunk = F.mse_loss(x_rec, target, reduction='none').mean(dim=(1, 2))
                
                all_losses.append(loss_chunk)
            
            all_losses = torch.cat(all_losses)
            all_losses = all_losses.view(B, self.num_eval_repeat)
            anomaly_score = all_losses.mean(dim=1)

            if return_dict:
                return {'reconstruction_loss': anomaly_score}
            return anomaly_score