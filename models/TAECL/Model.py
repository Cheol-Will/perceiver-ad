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
                    z, attn_enc = self.encoder(x_input)
                z = z.float()
            else:
                z, attn_enc = self.encoder(x_input)
            all_keys.append(z.cpu())

        all_keys = torch.cat(all_keys, dim=0).to(device)
        eval_memory = F.normalize(all_keys, dim=-1)
        self.memory_bank = eval_memory
        del all_keys
        torch.cuda.empty_cache()

    def forward(self, x):
        batch_size = x.shape[0]
        z, attn_enc = self.encoder(x)
        x_hat, attn_dec = self.decoder(z, self.pos_encoding)
        if not self.training:
            x = x.float() # turn off amp
            x_hat = x_hat.float()        
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
                'attn_enc': attn_enc,
                'attn_dec': attn_dec,
            }
            return output
        
    @torch.no_grad()
    def forward_knn(self, x):
        """
        Run KNN in latent space.
        """
        self.eval()
        z, attn_enc = self.encoder(x)
        bank = self.memory_bank
        ab = torch.einsum("bd,nd->bn", z, bank)
        aa = torch.einsum("bd,bd->b", z, z).unsqueeze(1)
        bb = torch.einsum("nd,nd->n", bank, bank).unsqueeze(0)
        dist = (aa + bb - 2.0 * ab).clamp_min_(0.0)
        top_k_list = [1, 5, 10, 16, 32, 64]
        scores = {}
        nn_dists, _ = torch.topk(dist, k=min(max(top_k_list), dist.shape[1]), dim=1, largest=False)
        for top_k in top_k_list:
            kk = min(top_k, nn_dists.shape[1])
            scores[f"knn{top_k}"] = nn_dists[:, :kk].mean(dim=1)

        return {
            "scores": scores,
            "nn_dists": nn_dists,
        }

    @torch.no_grad()
    def build_eval_attn_bank(self, train_loader, device, use_amp=False):
        self.eval()
        all_attns = []
        for (x_input, y_label) in train_loader:
            x_input = x_input.to(device)
            if use_amp:
                with autocast():
                    z, attn_enc = self.encoder(x_input)
                z = z.float()
            else:
                z, attn_enc = self.encoder(x_input)
            if isinstance(attn_enc, (list, tuple)):
                all_attns.append(torch.stack([a.detach().cpu() for a in attn_enc], dim=0))
            else:
                all_attns.append(attn_enc.detach().cpu())

        self.attn_bank = torch.cat(all_attns, dim=1).permute(1, 0, 2, 3, 4).contiguous()

        del all_attns
        torch.cuda.empty_cache()

    @torch.no_grad()
    def empty_eval_attn_bank(self,):
        self.attn_bank = None


    @torch.no_grad()
    def forward_knn_attn(self, x):
        self.eval()
        z, attn_enc = self.encoder(x)
        if isinstance(attn_enc, (list, tuple)):
            attn_enc = torch.stack([a.detach().cpu() for a in attn_enc], dim=0)
        else:
            attn_enc = attn_enc.detach().cpu()
        attn_enc = attn_enc.permute(1, 0, 2, 3, 4).contiguous()

        bank = self.attn_bank

        ab = torch.einsum("blhtt,nlhtt->bn", attn_enc, bank) # (B, N)
        aa = torch.einsum("blhtt,blhtt->b", attn_enc, attn_enc).unsqueeze(1) # (B, 1)
        bb = torch.einsum("nlhtt,nlhtt->n", bank, bank).unsqueeze(0) # (1, N)
        dist = (aa + bb - 2.0 * ab).clamp_min_(0.0)

        top_k_list = [1, 5, 10, 16, 32, 64]
        scores = {}
        nn_dists, _ = torch.topk(dist, k=min(max(top_k_list), dist.shape[1]), dim=1, largest=False)
        for top_k in top_k_list:
            kk = min(top_k, nn_dists.shape[1])
            scores[f"knn_attn{top_k}"] = nn_dists[:, :kk].mean(dim=1)

        return {
            "scores": scores,
            "nn_dists": nn_dists,
        }
    
    @torch.no_grad()
    def forward_knn_attn_cls(self, x):
        """
        Run KNN with attention weights of <CLS> token.
        """
        self.eval()
        _, attn_enc = self.encoder(x)
        if isinstance(attn_enc, (list, tuple)):
            attn_enc = torch.stack([a.detach().cpu() for a in attn_enc], dim=0)
        else:
            attn_enc = attn_enc.detach().cpu()
        # B, L, H, F, F
        attn_enc = attn_enc.permute(1, 0, 2, 3, 4).contiguous()
        attn_enc_cls = attn_enc[:, :, :, 0, :] # (B, L, H, F)
        bank_cls = self.attn_bank[:, :, :, 0, :] # # (N, L, H, F)
        
        ab = torch.einsum("blhf,nlhf->bn", attn_enc_cls, bank_cls) # (B, N)
        aa = torch.einsum("blhf,blhf->b", attn_enc_cls, attn_enc_cls).unsqueeze(1) # (B, 1)
        bb = torch.einsum("nlhf,nlhf->n", bank_cls, bank_cls).unsqueeze(0) # (1, N)
        dist = (aa + bb - 2.0 * ab).clamp_min_(0.0) # (B, N)

        top_k_list = [1, 5, 10, 16, 32, 64]
        scores = {}
        nn_dists, _ = torch.topk(dist, k=min(max(top_k_list), dist.shape[1]), dim=1, largest=False)
        for top_k in top_k_list:
            kk = min(top_k, nn_dists.shape[1])
            scores[f"knn_attn_cls{top_k}"] = nn_dists[:, :kk].mean(dim=1)

        return {
            "scores": scores,
            "nn_dists": nn_dists,
        }

    @torch.no_grad()
    def forward_combined(
        self, 
        x, 
        keyword: str = None,
        weight_list = None
    ):
        assert keyword is not None
        if keyword == 'knn':
            knn_scores = self.forward_knn(x)
        elif keyword == 'knn_attn':
            knn_scores = self.forward_knn_attn(x)
        elif keyword == 'knn_attn_cls':
            knn_scores = self.forward_knn_attn_cls(x)
        else:
            raise ValueError(f"Unknown keyword: {keyword}")

        if weight_list is None:
            weight_list = [0.01, 0.1, 1.0]

        device = x.device
        z, attn_enc = self.encoder(x)
        x_hat, attn_dec = self.decoder(z, self.pos_encoding)
        reconstruction_loss = F.mse_loss(x, x_hat, reduction='none').mean(dim=-1)
        
        knn_scores = knn_scores['scores']
        scores = {}
        for name, knn_score in knn_scores.items():
            for weight in weight_list:
                combined = reconstruction_loss + weight * knn_score.to(device)
                scores[f'comb_{name}_w{weight}'] = combined
        return {
            "combined": scores,
            "knn_scores": knn_scores,
        }