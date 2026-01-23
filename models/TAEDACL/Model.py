# Model.py

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from models.layers import BaseEncoder, BaseDecoder


class TAEDACL(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout_prob,
        temperature,
        byol_loss_weight,
        use_flash_attn: bool = False,
        depth_enc: int = None,
        depth_dec: int = None,
        dacl_alpha: float = 0.9,
        projector_dim: int = None,
        byol_momentum: float = 0.99,
        use_bn: bool = False,
    ):
        super().__init__()
        depth_enc = depth if depth_enc is None else depth_enc
        depth_dec = 1 if depth_dec is None else depth_dec
        projector_dim = hidden_dim if projector_dim is None else projector_dim

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

        self.temperature = float(temperature)
        self.byol_loss_weight = float(byol_loss_weight)

        self.dacl_alpha = float(dacl_alpha)
        self.byol_momentum = float(byol_momentum)
        self.use_bn = bool(use_bn)

        self.projector = self._make_mlp(hidden_dim, projector_dim, projector_dim, use_bn=self.use_bn)
        self.predictor = self._make_mlp(projector_dim, projector_dim, projector_dim, use_bn=self.use_bn)

        self.encoder_t = copy.deepcopy(self.encoder)
        self.projector_t = copy.deepcopy(self.projector)
        for p in self.encoder_t.parameters():
            p.requires_grad_(False)
        for p in self.projector_t.parameters():
            p.requires_grad_(False)

        self.reset_parameters()

    def _make_mlp(self, in_dim, hid_dim, out_dim, use_bn: bool):
        if use_bn:
            norm1 = nn.BatchNorm1d(hid_dim)
        else:
            norm1 = nn.LayerNorm(hid_dim)
        return nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            norm1,
            nn.GELU(),
            nn.Linear(hid_dim, out_dim),
        )

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        self._init_mlp(self.projector)
        self._init_mlp(self.predictor)
        self._sync_target(hard=True)

    def _init_mlp(self, mlp: nn.Module):
        for m in mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _sync_target(self, hard: bool = False, m: float = None):
        if hard:
            self.encoder_t.load_state_dict(self.encoder.state_dict())
            self.projector_t.load_state_dict(self.projector.state_dict())
            return

        momentum = self.byol_momentum if m is None else float(m)
        for p_o, p_t in zip(self.encoder.parameters(), self.encoder_t.parameters()):
            p_t.data.mul_(momentum).add_(p_o.data, alpha=1.0 - momentum)
        for p_o, p_t in zip(self.projector.parameters(), self.projector_t.parameters()):
            p_t.data.mul_(momentum).add_(p_o.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def update_target_network(self, m: float = None):
        self._sync_target(hard=False, m=m)

    @torch.no_grad()
    def empty_eval_memory_bank(self):
        self.memory_bank = None

    @torch.no_grad()
    def build_eval_memory_bank(self, train_loader, device, use_amp=False):
        self.eval()
        all_keys = []
        for (x_input, _) in train_loader:
            x_input = x_input.to(device)
            if use_amp:
                with autocast():
                    z, _ = self.encoder(x_input)
                z = z.float()
            else:
                z, _ = self.encoder(x_input)
            all_keys.append(z.cpu())

        all_keys = torch.cat(all_keys, dim=0).to(device)
        self.memory_bank = F.normalize(all_keys, dim=-1)
        del all_keys
        torch.cuda.empty_cache()

    def _dacl_views(self, x):
        b = x.shape[0]
        if b <= 1:
            return x, x

        device = x.device
        perm1 = torch.randperm(b, device=device)
        perm2 = torch.randperm(b, device=device)

        a = self.dacl_alpha
        lam1 = torch.empty(b, 1, device=device).uniform_(a, 1.0)
        lam2 = torch.empty(b, 1, device=device).uniform_(a, 1.0)

        v1 = lam1 * x + (1.0 - lam1) * x[perm1]
        v2 = lam2 * x + (1.0 - lam2) * x[perm2]
        return v1, v2

    def _byol_loss_per_sample(self, p, y):
        p = F.normalize(p, dim=-1)
        y = F.normalize(y, dim=-1)
        return (p - y).pow(2).mean(dim=-1)

    def forward(self, x):
        batch_size = x.shape[0]

        z, attn_enc = self.encoder(x)
        x_hat, attn_dec = self.decoder(z, self.pos_encoding)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=-1)

        if self.training:
            v1, v2 = self._dacl_views(x)

            z1, _ = self.encoder(v1)
            z2, _ = self.encoder(v2)

            y1 = self.projector(z1)
            y2 = self.projector(z2)

            p1 = self.predictor(y1)
            p2 = self.predictor(y2)

            with torch.no_grad():
                z1_t, _ = self.encoder_t(v1)
                z2_t, _ = self.encoder_t(v2)
                y1_t = self.projector_t(z1_t)
                y2_t = self.projector_t(z2_t)

            byol_loss = self._byol_loss_per_sample(p1, y2_t) + self._byol_loss_per_sample(p2, y1_t)
            ssl_loss = self.byol_loss_weight * byol_loss
            loss = reconstruction_loss + ssl_loss

            output = {
                'loss': loss,
                'reconstruction_loss': reconstruction_loss,
                'byol_loss': byol_loss,
                'ssl_loss': ssl_loss,
            }
            return output
        else:
            z = z.float()
            z_norm = F.normalize(z, dim=-1)

            if self.memory_bank is None:
                score = torch.zeros(batch_size, device=x.device, dtype=z_norm.dtype)
            else:
                logits = torch.matmul(z_norm, self.memory_bank.T) / self.temperature
                score = -torch.logsumexp(logits, dim=-1) * self.byol_loss_weight

            output = {
                'reconstruction_loss': reconstruction_loss,
                'byol_score': score,
                'combined': reconstruction_loss + score,
                'latent': z_norm,
                'x_hat': x_hat,
                'attn_enc': attn_enc,
                'attn_dec': attn_dec,
            }
            return output
