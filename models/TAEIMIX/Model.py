import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.distributions import Beta
from models.layers import BaseEncoder, BaseDecoder


class TAEIMIX(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout_prob,
        temperature,
        imix_loss_weight,
        use_flash_attn: bool = False,
        depth_enc: int = None,
        depth_dec: int = None,
        mixup_alpha: float = 1.0,
        projector_dim: int = None,
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
        self.imix_loss_weight = float(imix_loss_weight)
        self.mixup_alpha = float(mixup_alpha)
        self.use_bn = bool(use_bn)

        self.projector = self._make_mlp(hidden_dim, projector_dim, projector_dim, use_bn=self.use_bn)
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

    def _init_mlp(self, mlp: nn.Module):
        for m in mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _sample_lam(self, b: int, device):
        if self.mixup_alpha <= 0.0:
            return torch.ones(b, 1, device=device)
        dist = Beta(self.mixup_alpha, self.mixup_alpha)
        return dist.sample((b, 1)).to(device)

    def _make_views(self, x):
        b = x.shape[0]
        if b <= 1:
            return x, x

        device = x.device
        perm1 = torch.randperm(b, device=device)
        perm2 = torch.randperm(b, device=device)

        lam1 = self._sample_lam(b, device)
        lam2 = self._sample_lam(b, device)

        v1 = lam1 * x + (1.0 - lam1) * x[perm1]
        v2 = lam2 * x + (1.0 - lam2) * x[perm2]
        return v1, v2

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

    def _imix_loss_per_sample(self, v1, v2):
        b = v1.shape[0]
        if b <= 1:
            return torch.zeros(b, device=v1.device, dtype=v1.dtype)

        device = v1.device
        perm = torch.randperm(b, device=device)
        lam = self._sample_lam(b, device)

        v1_mix = lam * v1 + (1.0 - lam) * v1[perm]

        z_mix, _ = self.encoder(v1_mix)
        z2, _ = self.encoder(v2)

        y_mix = F.normalize(self.projector(z_mix), dim=-1)
        y2 = F.normalize(self.projector(z2), dim=-1)

        logits = torch.matmul(y_mix, y2.transpose(0, 1)) / self.temperature

        t0 = torch.arange(b, device=device)
        ce0 = F.cross_entropy(logits, t0, reduction="none")
        ce1 = F.cross_entropy(logits, perm, reduction="none")

        lam_flat = lam.squeeze(-1).to(dtype=ce0.dtype)
        return lam_flat * ce0 + (1.0 - lam_flat) * ce1

    def forward(self, x):
        batch_size = x.shape[0]

        z, attn_enc = self.encoder(x)
        x_hat, attn_dec = self.decoder(z, self.pos_encoding)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

        if self.training:
            v1, v2 = self._make_views(x)
            imix_loss = self._imix_loss_per_sample(v1, v2)
            ssl_loss = self.imix_loss_weight * imix_loss
            loss = reconstruction_loss + ssl_loss

            return {
                "loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "imix_loss": imix_loss,
                "ssl_loss": ssl_loss,
            }

        z = z.float()
        z_norm = F.normalize(z, dim=-1)

        if self.memory_bank is None:
            score = torch.zeros(batch_size, device=x.device, dtype=z_norm.dtype)
        else:
            logits = torch.matmul(z_norm, self.memory_bank.T) / self.temperature
            score = -torch.logsumexp(logits, dim=-1) * self.imix_loss_weight

        return {
            "reconstruction_loss": reconstruction_loss,
            "imix_score": score,
            "combined": reconstruction_loss + score,
            "latent": z_norm,
            "x_hat": x_hat,
            "attn_enc": attn_enc,
            "attn_dec": attn_dec,
        }
