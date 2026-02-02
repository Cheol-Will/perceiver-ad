import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import BaseEncoder, BaseDecoder


class RECL(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout_prob,
        temperature,
        dacl_alpha,
        dacl_beta,
        contra_loss_weight,
        consis_loss_weight,
        depth_dec: int = None,
        depth_enc: int = None,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        depth_enc = depth if depth_enc is None else depth_enc
        depth_dec = 1 if depth_dec is None else depth_dec
        projector_dim = hidden_dim

        self.encoder = BaseEncoder(
            num_features, hidden_dim, depth_enc, num_heads,
            mlp_ratio, dropout_prob, use_flash_attn
        )
        self.decoder = BaseDecoder(
            num_features, hidden_dim, depth_dec, num_heads,
            mlp_ratio, dropout_prob, use_flash_attn
        )
        self.projector = self._make_mlp(hidden_dim, projector_dim, projector_dim)
        self.pos_encoding = nn.Parameter(torch.empty(1, num_features, hidden_dim))

        self.num_features = num_features
        self.temperature = float(temperature)
        self.dacl_alpha = float(dacl_alpha)
        self.dacl_beta = float(dacl_beta)
        self.contra_loss_weight = contra_loss_weight
        self.consis_loss_weight = consis_loss_weight

        self.reset_parameters()

    def _make_mlp(self, in_dim, hidden_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        self._init_mlp(self.projector)

    def _init_mlp(self, mlp: nn.Module):
        for m in mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _dacl_views(self, x):
        b = x.shape[0]
        if b <= 1:
            return x

        device = x.device
        perm = torch.randperm(b, device=device)

        a = self.dacl_alpha
        lam = torch.empty(b, 1, device=device).uniform_(a, 1.0)
        x_aug = lam * x + (1.0 - lam) * x[perm]
        return x_aug

    def _latent_mix(self, z):
        b = z.shape[0]
        if b <= 1:
            return z

        perm = torch.randperm(b, device=z.device)
        a = self.dacl_beta

        lam = torch.empty(b, 1, device=z.device).uniform_(a, 1.0)  # (B,1)
        return lam * z + (1.0 - lam) * z[perm]

    def simclr_loss(self, p1, p2):
        # (B, D)
        batch_size = p1.shape[0]
        p = torch.cat([p1, p2], dim=0)
        logits = torch.matmul(p, p.T) / self.temperature # (2B, 2B)
        logits.fill_diagonal_(-float("inf")) # exclude self-logit

        labels = torch.arange(2 * batch_size).to(p1.device)
        labels = (labels + batch_size) % (2 * batch_size) # (B, ..., 2B-1, 0, ..., B-1)

        contra_loss = F.cross_entropy(logits, labels, reduction='none')
        return contra_loss

    def forward(self, x):

        z, attn_enc = self.encoder(x)
        
        if self.training:
            z_mix = self._latent_mix(z)
            x_hat, _ = self.decoder(z_mix, self.pos_encoding)
            recon_loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

            x_aug = self._dacl_views(x) 
            z_aug, _ = self.encoder(x_aug)
            
            p = self.projector(z)
            p_aug = self.projector(z_aug)

            p_norm = F.normalize(p, dim=-1)
            p_aug_norm = F.normalize(p_aug, dim=-1)

            contra_loss = self.simclr_loss(p_norm, p_aug_norm)
            contra_loss = contra_loss * self.contra_loss_weight

            loss = recon_loss.mean() + contra_loss.mean()

            z_hat, _ = self.encoder(x_hat)
            p_hat = self.projector(z_hat)
            p_hat_norm = F.normalize(p_hat, dim=-1)
            consis_loss= self.simclr_loss(p_norm, p_hat_norm)
            consis_loss = consis_loss * self.consis_loss_weight
            loss += consis_loss.mean()

            return {
                "loss": loss,
                "recon_loss": recon_loss.mean(),
                "contra_loss": contra_loss.mean(),
                "consis_loss": consis_loss.mean(),
            }
        else:
            x_hat, attn_dec = self.decoder(z, self.pos_encoding)
            x = x.float()
            x_hat = x_hat.float()
            recon_loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1) # (B, )
            z_norm = F.normalize(z.float(), dim=-1) # for analysis

            return {
                "recon_loss": recon_loss,
                "latent": z_norm,
                "x_hat": x_hat,
                "attn_enc": attn_enc,
                "attn_dec": attn_dec,
            }