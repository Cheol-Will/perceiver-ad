import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from models.layers import BaseEncoder, BaseDecoder


class TAEML(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout_prob,
        dacl_beta: float = None,
        use_flash_attn: bool = False,
        depth_dec: int = None,
        depth_enc: int = None,
        use_bn: bool = False,
        use_swap: bool = False,
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

        self.pos_encoding = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        self.memory_bank = None

        self.num_features = num_features
        self.dacl_beta = float(dacl_beta)
        self.use_bn = bool(use_bn)
        self.use_swap = use_swap

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def _latent_mix(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        if b <= 1:
            return z

        perm = torch.randperm(b, device=z.device)
        a = self.dacl_beta

        if not self.use_swap:
            lam = torch.empty(b, 1, device=z.device).uniform_(a, 1.0)  # (B,1)
            return lam * z + (1.0 - lam) * z[perm]
        else:
            d = z.shape[1]
            mask = torch.empty(b, d, device=z.device).bernoulli_(a)      # (B,D)
            return mask * z + (1.0 - mask) * z[perm]

    def forward(self, x):
        batch_size = x.shape[0]

        z, _ = self.encoder(x)

        if self.training:
            z_mix = self._latent_mix(z)
            x_hat, _ = self.decoder(z_mix, self.pos_encoding)
            recon_loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

            loss = recon_loss.mean()
            return {
                "loss": loss,
                "recon_loss": recon_loss.mean(),
            }

        z, attn_enc = self.encoder(x)
        x_hat, attn_dec = self.decoder(z, self.pos_encoding)
        x = x.float()
        x_hat = x_hat.float()
        recon_loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

        return {
            "recon_loss": recon_loss,
            "latent": z,
            "x_hat": x_hat,
            "attn_enc": attn_enc,
            "attn_dec": attn_dec,
        }