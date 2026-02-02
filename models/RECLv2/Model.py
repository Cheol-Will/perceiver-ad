import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import BaseEncoder, BaseDecoder


class RECLv2(nn.Module):
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
        depth_dec: int = None,
        depth_enc: int = None,
        use_flash_attn: bool = False,
    ):  
        """
        Recon + Contra (merged)
        """
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

    def triplet_contra_loss(self, p: torch.Tensor) -> torch.Tensor:
        assert p.dim() == 2
        N = p.shape[0]
        assert N % 3 == 0
        B = N // 3
        device = p.device

        logits = (p @ p.T) / self.temperature # (N, N)
        logits = logits.float() # turn off amp
        logits.fill_diagonal_(-1e9)

        origin_id = torch.arange(B, device=device).repeat(3)  # (N, ): same source 
        exclude_id = torch.eye(N, dtype=torch.bool, device=device) # exclue self-positive
        labels = (origin_id[:, None] == origin_id[None, :]) & (~exclude_id)  # (N, N)

        target = labels.float()
        target = target / target.sum(dim=1, keepdim=True) # assign 0.5 for each positive pair.

        loss = F.cross_entropy(logits, target, reduction="mean")
        return loss


    def forward(self, x):

        z_ori, attn_enc = self.encoder(x)
        
        if self.training:
            z_mix = self._latent_mix(z_ori)
            x_hat, _ = self.decoder(z_mix, self.pos_encoding)

            x_aug = self._dacl_views(x)
            z_aug, _ = self.encoder(x_aug)
            
            z_merged = torch.cat([z_ori, z_mix, z_aug], dim=0)
            p_merged = self.projector(z_merged)
            p_merged = F.normalize(p_merged, dim=-1)

            contra_loss = self.contra_loss_weight * self.triplet_contra_loss(p_merged)
            recon_loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)
            loss = recon_loss.mean() + contra_loss

            return {
                "loss": loss,
                "recon_loss": recon_loss.mean(),
                "contra_loss": contra_loss,
            }
        else:
            x_hat, attn_dec = self.decoder(z_ori, self.pos_encoding)
            x = x.float()
            x_hat = x_hat.float()
            recon_loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1) # (B, )
            z_norm = F.normalize(z_ori.float(), dim=-1) # for analysis

            return {
                "recon_loss": recon_loss,
                "latent": z_norm,
                "x_hat": x_hat,
                "attn_enc": attn_enc,
                "attn_dec": attn_dec,
            }