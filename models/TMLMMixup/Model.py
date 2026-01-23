import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from models.layers import BaseDecoder, BaseEncoder


class TMLMMixup(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout_prob,
        use_flash_attn: bool = False,
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.5,
    ):
        super().__init__()
        self.encoder = BaseEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_prob=dropout_prob,
            use_flash_attn=use_flash_attn,
        )
        self.decoder = BaseDecoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_prob=dropout_prob,
        )
        self.pos_encoding = nn.Parameter(torch.empty(1, num_features, hidden_dim))

        self.mixup_alpha = float(mixup_alpha)
        self.mixup_prob = float(mixup_prob)
        self._beta = Beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def _mixup(self, x):
        """
        Apply mixup with a fixed probability and lambda.
        """

        if self._beta is None or self.mixup_prob <= 0 or x.size(0) < 2:
            return x

        if torch.rand(1, device=x.device).item() >= self.mixup_prob:
            return x

        b = x.size(0)
        perm = torch.randperm(b, device=x.device)

        lam = self._beta.sample((b, 1)).to(x.device).type_as(x)
        lam = torch.maximum(lam, 1.0 - lam)

        return lam * x + (1.0 - lam) * x[perm]

    def forward(self, x, return_dict: bool = False):
        x = x.float()

        if self.training:
            x_in = self._mixup(x)
            z, _ = self.encoder(x_in)
            x_hat, _ = self.decoder(z, self.pos_encoding)

            rec_loss = F.mse_loss(x_hat.float(), x, reduction="none").mean(dim=-1)

            if return_dict:
                return {"reconstruction_loss": rec_loss, "latent": z, "x_hat": x_hat}
            return rec_loss
        else:
            z, _ = self.encoder(x)
            x_hat, _ = self.decoder(z, self.pos_encoding)

            rec_loss = F.mse_loss(x_hat.float(), x, reduction="none").mean(dim=-1)

            if return_dict:
                return {"reconstruction_loss": rec_loss, "latent": z, "x_hat": x_hat}
            return rec_loss
