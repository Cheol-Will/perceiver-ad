import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from models.layers import BaseEncoder, BaseDecoder


class TAEDACLv4(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout_prob,
        temperature,
        contra_loss_weight,
        use_flash_attn: bool = False,
        depth_dec: int = None,
        depth_enc: int = None,
        dacl_alpha: float = 0.9,
        dacl_beta: float = None,
        use_bn: bool = False,
        use_swap: bool = False,
    ):
        super().__init__()
        depth_enc = depth if depth_enc is None else depth_enc
        depth_dec = 1 if depth_dec is None else depth_dec
        dacl_beta = dacl_alpha if dacl_beta is None else dacl_beta
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
        self.temperature = float(temperature)
        self.contra_loss_weight = contra_loss_weight
        self.dacl_alpha = float(dacl_alpha)
        self.dacl_beta = float(dacl_beta)
        self.use_bn = bool(use_bn)
        self.use_swap = use_swap

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
    
    def _swap_views(self, x):
        b = x.shape[0]
        if b <= 1:
            return x

        device = x.device
        perm = torch.randperm(b, device=device)

        a = self.dacl_alpha
        lam = torch.empty(b, self.num_features, device=device).bernoulli_(a) # 
        x_aug = lam * x + (1.0 - lam) * x[perm]
        return x_aug
    
    def simclr_loss(self, p1, p2):
        # (B, D)
        batch_size = p1.shape[0]
        p = torch.cat([p1, p2], dim=0)
        logits = torch.matmul(p, p.T) / self.temperature # (2B, 2B)
        logits.fill_diagonal_(-float("inf")) # exclude diagonal

        labels = torch.arange(2 * batch_size).to(p1.device)
        labels = (labels + batch_size) % (2 * batch_size) # (B, ..., 2B-1, 0, ..., B-1)

        contra_loss = F.cross_entropy(logits, labels, reduction='none')
        return contra_loss

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

            x_aug = self._dacl_views(x) if not self.use_swap else self._swap_views(x)
            z_aug, _ = self.encoder(x_aug)

            p = self.projector(z)
            p_aug = self.projector(z_aug)

            p_norm = F.normalize(p, dim=-1)
            p_aug_norm = F.normalize(p_aug, dim=-1)

            contra_loss = self.simclr_loss(p_norm, p_aug_norm)
            contra_loss = contra_loss * self.contra_loss_weight

            loss = recon_loss.mean() + contra_loss.mean()
            return {
                "loss": loss,
                "recon_loss": recon_loss.mean(),
                "contra_loss": contra_loss.mean(),
            }

        z, attn_enc = self.encoder(x)
        x_hat, attn_dec = self.decoder(z, self.pos_encoding)

        x = x.float()
        x_hat = x_hat.float()
        recon_loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

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
            "latent": z_norm,
            "x_hat": x_hat,
            "attn_enc": attn_enc,
            "attn_dec": attn_dec,
        }

        
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

    @torch.no_grad()
    def forward_knn(self, x):
        """
        Run KNN in latent space.
        """
        self.eval()
        z, attn_enc = self.encoder(x)
        z = F.normalize(z, dim=-1)
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
    def build_eval_attn_bank(
        self, 
        train_loader, 
        device, 
        use_amp=False,
    ):
        self.eval()
        all_attns = []
        for (x_input, y_label) in train_loader:
            x_input = x_input.to(device)
            if use_amp:
                with autocast():
                    z, attn_enc = self.encoder(x_input)
            else:
                z, attn_enc = self.encoder(x_input)

            # attn_enc: (depth, batch, head, token, token)                
            if isinstance(attn_enc, (list, tuple)):
                stacked = torch.stack([a.detach().cpu() for a in attn_enc], dim=0)
                all_attns.append(stacked)
            else:
                # depth = 1 case
                all_attns.append(attn_enc.detach().cpu())

        self.attn_bank = torch.cat(all_attns, dim=1).permute(1, 0, 2, 3, 4).contiguous()

        del all_attns
        torch.cuda.empty_cache()

    @torch.no_grad()
    def empty_eval_attn_bank(self,):
        self.attn_bank = None

    @torch.no_grad()
    def forward_knn_attn(
        self,
        x,
        use_cls=False,
        use_first=False,
        use_penul=False,
    ):
        prefix = "knn_attn_cls" if use_cls else "knn_attn"
        if use_penul:
            prefix += "_penul"
        if use_first:
            prefix += "_first"

        self.eval()
        _, attn_enc = self.encoder(x)

        if isinstance(attn_enc, (list, tuple)):
            L = len(attn_enc)
            if use_penul:
                idx = -2 if L >= 2 else -1
            elif use_first:
                idx = 0
            else:
                idx = None

            if idx is None:
                attn_enc = torch.stack(attn_enc, dim=0) # (L,B,H,F,F)
            else:
                attn_enc = attn_enc[idx].unsqueeze(0) # (1,B,H,F,F)
        
        if use_cls:
            attn_enc = attn_enc[:, :, :, 0, :].unsqueeze(3) # (L,B,H,1,F)

        attn_enc = attn_enc.detach().to("cpu", non_blocking=True)
        attn_enc = attn_enc.permute(1, 0, 2, 3, 4).contiguous()  # (B,L,H,*,F)

        bank = self.attn_bank  # (N,L,H,F,F)
        if use_penul:
            bank = bank[:, -1:, :, :, :]  # (N,1,H,F,F)
        elif use_first:
            bank = bank[:, 0:1, :, :, :]  # (N,1,H,F,F)
        if use_cls:
            bank = bank[:, :, :, 0, :].unsqueeze(3)  # (N,L,H,1,T)

        ab = torch.einsum("blhij,nlhij->bn", attn_enc, bank)
        aa = torch.einsum("blhij,blhij->b", attn_enc, attn_enc).unsqueeze(1)
        bb = torch.einsum("nlhij,nlhij->n", bank, bank).unsqueeze(0)
        dist = (aa + bb - 2.0 * ab).clamp_min_(0.0)

        top_k_list = [1, 5, 10, 16, 32, 64]
        Kmax = min(max(top_k_list), dist.shape[1])
        nn_dists, _ = torch.topk(dist, k=Kmax, dim=1, largest=False)

        scores = {}
        for k in top_k_list:
            kk = min(k, Kmax)
            scores[f"{prefix}{k}"] = nn_dists[:, :kk].mean(dim=1)

        return {"scores": scores, "nn_dists": nn_dists}
    
    @torch.no_grad()
    def forward_combined(
        self, 
        x, 
        keyword: str = None,
        weight_list = None,
    ):
        assert keyword is not None
        if keyword == 'knn':
            knn_scores = self.forward_knn(x)
        elif 'knn_attn' in keyword:
            use_cls = 'cls' in keyword
            use_penul = 'penul' in keyword
            use_first = 'first' in keyword
            knn_scores = self.forward_knn_attn(
                x, 
                use_cls=use_cls,
                use_penul=use_penul,
                use_first=use_first,
            )
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
    
    @torch.no_grad()
    def forward_retrieval(self, x, top_k_list=(1, 5, 10, 16, 32, 64), pool="kth"):
        self.eval()

        bank = self.memory_bank
        if bank is None:
            raise RuntimeError("memory_bank is None. Call build_eval_memory_bank() first.")

        z, attn_enc = self.encoder(x)
        z = F.normalize(z.float(), dim=-1)
        x_hat, attn_dec = self.decoder(z, self.pos_encoding)
        recon = F.mse_loss(x_hat.float(), x.float(), reduction='none').mean(dim=-1)

        bank = bank.to(z.device, non_blocking=True)
        sim = torch.matmul(z, bank.T)  # reverse order with L2

        Kmax = min(max(top_k_list), sim.shape[1])
        nn_sim, nn_idx = torch.topk(sim, k=Kmax, dim=1, largest=True)  # nearest = largest sim

        scores = {}
        recons = {}
        retrieved_latents = {}

        for k in top_k_list:
            kk = min(k, Kmax)
            idx_k = nn_idx[:, :kk]
            lat_k = bank[idx_k]

            if pool == "mean":
                z_ret = lat_k.mean(dim=1)
            elif pool == "kth":
                z_ret = lat_k[:, -1, :]
            else:
                raise ValueError(f"Unknown pool: {pool}")

            x_hat_k, _ = self.decoder(z_ret, self.pos_encoding)
            recon_k = F.mse_loss(x_hat_k.float(), x.float(), reduction="none").mean(dim=-1)

            key = f"ret_{pool}_top{k}"
            scores[key] = recon_k
            recons[key] = x_hat_k
            scores["comb_" + key] = recon + recon_k
            retrieved_latents[key] = z_ret

        return {
            "scores": scores,
            "nn_idx": nn_idx,
            "nn_sim": nn_sim,
            "retrieved_latents": retrieved_latents,
            "x_hat": recons,
        }
    
    @torch.no_grad()
    def forward_repeat(self, x, max_n = 5):
        self.eval()
        scores = {}
        x_hat = x
        cum = None
        for i in range(1, max_n+1):
            z, _ = self.encoder(x_hat)
            x_hat, _ = self.decoder(z, self.pos_encoding)
            recon_loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=-1)
            
            cum = cum + recon_loss if cum is not None else recon_loss
            scores[f"{i}th_recon_score"] = cum

        return {
            "scores": scores,
        }