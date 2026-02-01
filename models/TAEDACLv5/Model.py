import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from models.TAEDACLv4.Model import TAEDACLv4


class TAEDACLv5(TAEDACLv4):
    def __init__(
        self,
        **keyward,
    ):
        super().__init__(**keyward)

    def forward(self, x):
        batch_size = x.shape[0]

        z, attn_enc = self.encoder(x)
        z = self.latent_bn(z) if self.use_bn else z
        
        if self.training:
            z_mix = self._latent_mix(z)
            x_hat, _ = self.decoder(z_mix, self.pos_encoding)
            recon_loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

            x_aug = self._dacl_views(x) if not self.use_swap else self._swap_views(x)
            z_aug, _ = self.encoder(x_aug)
            z_aug = self.latent_bn(z_aug) if self.use_bn else z_aug
            
            p = self.projector(z)
            p_aug = self.projector(z_aug)

            p_norm = F.normalize(p, dim=-1)
            p_aug_norm = F.normalize(p_aug, dim=-1)

            contra_loss = self.simclr_loss(p_norm, p_aug_norm)
            contra_loss = contra_loss * self.contra_loss_weight

            loss = recon_loss.mean() + contra_loss.mean()

            if self.cycle_loss_weight is not None:
                z_hat, _ = self.encoder(x_hat)
                cycle_loss = F.mse_loss(z_hat, z.detach(), reduction='none').mean(dim=1)
                cycle_loss = cycle_loss * self.cycle_loss_weight
                loss += cycle_loss.mean()
            else:
                cycle_loss = torch.zeros(batch_size, device=x.device, dtype=z.dtype)

            return {
                "loss": loss,
                "recon_loss": recon_loss.mean(),
                "contra_loss": contra_loss.mean(),
                "cycle_loss": cycle_loss.mean(),
            }

        x_hat, attn_dec = self.decoder(z, self.pos_encoding)
        x = x.float()
        x_hat = x_hat.float()
        recon_loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)
        
        # cycle loss
        if self.cycle_loss_weight is not None:
            z_hat, _ = self.encoder(x_hat)
            cycle_loss = F.mse_loss(z_hat, z, reduction='none').mean(dim=1)
            cycle_loss = cycle_loss * self.cycle_loss_weight
            recon_and_cycle_loss = recon_loss + cycle_loss
        else:
            recon_and_cycle_loss = torch.zeros(batch_size, device=x.device, dtype=z_norm.dtype)

        z_norm = F.normalize(z.float(), dim=-1)

        if self.memory_bank is None:
            score = torch.zeros(batch_size, device=x.device, dtype=z_norm.dtype)
        else:
            logits = torch.matmul(z_norm, self.memory_bank.T) / self.temperature
            score = -torch.logsumexp(logits, dim=-1) * self.contra_loss_weight

        return {
            "recon_loss": recon_loss,
            "contra_score": score,
            "combined": recon_loss + score,
            "recon_and_cycle_loss": recon_and_cycle_loss,
            "latent": z_norm,
            "x_hat": x_hat,
            "attn_enc": attn_enc,
            "attn_dec": attn_dec,
        }