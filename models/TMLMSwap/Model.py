import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import BaseDecoder, BaseEncoder


class TMLMSwap(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout_prob,
        use_flash_attn: bool = False,
        swap_ratio: float = 0.2,
        num_eval_repeat: int = 10,
        eval_chunk_size: int = 10,
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

        self.swap_ratio = float(swap_ratio)
        self.num_eval_repeat = int(num_eval_repeat)
        self.eval_chunk_size = int(eval_chunk_size)

        self.register_buffer("eval_swap_list", torch.empty(0, num_features, dtype=torch.long))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    @torch.no_grad()
    def _build_eval_swap_list(self, device: torch.device):
        R = self.num_eval_repeat
        F_ = self.pos_encoding.shape[1]
        swap_list = []
        for _ in range(R):
            idx = self._sample_swap_index(batch_size=1, num_features=F_, device=device)
            swap_list.append(idx[0])
        self.eval_swap_list = torch.stack(swap_list, dim=0)  # (R, F)

    def _sample_swap_index(self, batch_size: int, num_features: int, device: torch.device):
        base = torch.arange(num_features, device=device).unsqueeze(0).expand(batch_size, -1).clone()

        k = int(round(num_features * self.swap_ratio))
        k = max(2, min(k, num_features))

        if k % 2 == 1:
            k = k - 1 if k > 2 else 2

        for b in range(batch_size):
            cols = torch.randperm(num_features, device=device)[:k]
            cols = cols.view(-1, 2)
            i = cols[:, 0]
            j = cols[:, 1]
            base[b, i] = j
            base[b, j] = i

        return base  # (B, F)

    def swap_cols(self, x: torch.Tensor):
        return self._sample_swap_index(
            batch_size=x.shape[0],
            num_features=x.shape[1],
            device=x.device,
        )

    def forward(self, x, return_dict: bool = False):
        if self.training:
            swap_index = self.swap_cols(x)
            x_aug = x.gather(1, swap_index)

            z, _ = self.encoder(x_aug)
            x_hat, _ = self.decoder(z, self.pos_encoding)

            reconstruction_loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

            if return_dict:
                return {
                    "reconstruction_loss": reconstruction_loss,
                    "latent": z,
                    "x_hat": x_hat,
                }
            return reconstruction_loss

        batch_size, num_features = x.shape
        R = self.num_eval_repeat
        chunk_size = self.eval_chunk_size

        if self.eval_swap_list.numel() == 0 or self.eval_swap_list.shape[0] != R:
            self._build_eval_swap_list(device=x.device)

        loss_list = []
        x_hat_list = []
        latent_list = []

        for start_idx in range(0, R, chunk_size):
            end_idx = min(start_idx + chunk_size, R)

            swap_chunk = self.eval_swap_list[start_idx:end_idx].to(x.device)  # (r, F)
            current_r = swap_chunk.shape[0]

            x_expanded = (
                x.unsqueeze(1)
                .expand(batch_size, current_r, num_features)
                .reshape(batch_size * current_r, num_features)
            )
            swap_expanded = (
                swap_chunk.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .reshape(batch_size * current_r, num_features)
            )

            x_aug = x_expanded.gather(1, swap_expanded)

            z_chunk, _ = self.encoder(x_aug)  # (B*r, D)
            x_hat_chunk, _ = self.decoder(z_chunk, self.pos_encoding)  # (B*r, F)

            loss_chunk = F.mse_loss(
                x_hat_chunk.float(), x_expanded.float(), reduction="none"
            ).mean(dim=-1)  # (B*r)

            loss_list.append(loss_chunk.view(batch_size, current_r))
            x_hat_list.append(x_hat_chunk.view(batch_size, current_r, num_features))
            if return_dict:
                latent_list.append(z_chunk.view(batch_size, current_r, -1))

        loss_all = torch.cat(loss_list, dim=1)  # (B, R)
        x_hat_all = torch.cat(x_hat_list, dim=1)  # (B, R, F)

        reconstruction_loss = loss_all.mean(dim=1)  # (B)
        x_hat = x_hat_all.mean(dim=1).float()  # (B, F)

        if return_dict:
            latent_all = (
                torch.cat(latent_list, dim=1).mean(dim=1) if len(latent_list) > 0 else None
            )
            return {
                "reconstruction_loss": reconstruction_loss,
                "latent": latent_all,
                "x_hat": x_hat,
            }
        return reconstruction_loss
