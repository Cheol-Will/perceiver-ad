import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class FeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int):
        super().__init__()
        self.embedding_matrix = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding_matrix)
        
    def forward(self, x: torch.Tensor, perm: Optional[torch.Tensor] = None) -> torch.Tensor:
        if perm is not None:
            # Permute both input and embedding
            x = x[:, perm]
            embedding = self.embedding_matrix[:, perm, :]
        else:
            embedding = self.embedding_matrix
            
        x_embedded = x.unsqueeze(-1) * embedding  # (B, F, D)
        return x_embedded


class RotaryPositionalEmbedding(nn.Module):
    """RoPE for better positional encoding (NEPA style)"""
    def __init__(self, dim: int, max_seq_len: int = 1024):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, D)
        return: (B, S, D) with rotary embedding applied
        """
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (S, D)
        
        cos_emb = emb.cos().unsqueeze(0)  # (1, S, D)
        sin_emb = emb.sin().unsqueeze(0)  # (1, S, D)
        
        # Rotate
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.cat([-x2, x1], dim=-1)
        
        return x * cos_emb + rotated * sin_emb


class CausalMultiHeadAttention(nn.Module):
    """Causal Multi-Head Attention with QK-Norm (NEPA style)"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        # QK-Norm for stability (NEPA style)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for module in [self.q, self.k, self.v, self.proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor, causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, S, D)
        causal: whether to use causal masking
        return: (B, S, D), (B, H, S, S) attention weights
        """
        B, S, D = x.shape
        
        q = self.q(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # QK-Norm
        q, k = self.q_norm(q), self.k_norm(k)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, S, S)
        
        # Causal mask
        if causal:
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        out = self.proj_drop(self.proj(out))
        
        return out, attn


class SwiGLU(nn.Module):
    """SwiGLU activation (NEPA style)"""
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Transformer block with LayerScale (NEPA style)"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        drop: float = 0.0,
        layer_scale_init: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalMultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            attn_drop=drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), drop)
        
        # LayerScale for stability
        self.layer_scale_1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(layer_scale_init * torch.ones(dim))
        
    def forward(self, x: torch.Tensor, causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weight = self.attn(self.norm1(x), causal=causal)
        x = x + self.layer_scale_1 * attn_out
        x = x + self.layer_scale_2 * self.mlp(self.norm2(x))
        return x, attn_weight


class NEPA_TAD(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        dropout_prob: float = 0.0,
        use_rope: bool = False,
        layer_scale_init: float = 1e-5,
        num_permutations: int = 50,  # For ensemble at test time
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_permutations = num_permutations
        self.use_rope = use_rope
        
        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(num_features, hidden_dim)
        
        self.pos_encoding = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        
        # RoPE (optional, NEPA style)
        if use_rope:
            self.rope = RotaryPositionalEmbedding(hidden_dim, max_seq_len=num_features)
        
        # Causal Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=dropout_prob,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(depth)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        print(f"NEPA-TAD initialized: features={num_features}, dim={hidden_dim}, "
              f"depth={depth}, heads={num_heads}, perms={num_permutations}")
        
    def _compute_nepa_loss(
        self, 
        z: torch.Tensor, 
        z_hat: torch.Tensor,
        reduction: str = 'none'
    ) -> torch.Tensor:
        # Shift 
        pred = z_hat[:, :-1, :] 
        target = z[:, 1:, :] 
        
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target.detach(), dim=-1)  # Stop gradient on target
        
        cos_sim = (pred_norm * target_norm).sum(dim=-1)  # (B, F-1)
        
        # so higher is worse
        loss_per_pos = 1 - cos_sim  # (B, F-1)
        
        if reduction == 'none':
            return loss_per_pos.mean(dim=-1)  # (B,)
        elif reduction == 'mean':
            return loss_per_pos.mean()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    def forward_single(
        self, 
        x: torch.Tensor, 
        perm: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        z = self.tokenizer(x, perm)  # (B, F, D)
        if perm is not None:
            pos = self.pos_encoding[:, perm, :]
        else:
            pos = self.pos_encoding
        z = z + pos
        
        if self.use_rope:
            z = self.rope(z)
        
        # Transformer blocks
        z_hat = z
        for block in self.blocks:
            z_hat, _ = block(z_hat, causal=causal)
        
        z_hat = self.norm(z_hat)
        
        # Compute loss
        loss = self._compute_nepa_loss(z, z_hat)
        
        return loss, z, z_hat
    
    def forward(
        self, 
        x: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor:
        """
        Training forward: random permutation each time
        
        x: (B, F)
        Returns: loss (B,)
        """
        B, F = x.shape
        
        # Random permutation for training
        perm = torch.randperm(F, device=x.device)
        
        loss, z, z_hat = self.forward_single(x, perm=perm, causal=True)
        
        if return_details:
            return loss, z, z_hat, perm
        return loss
    
    @torch.no_grad()
    def compute_anomaly_score(
        self, 
        x: torch.Tensor,
        num_permutations: Optional[int] = None,
        aggregation: str = 'mean',  
    ) -> torch.Tensor:

        if num_permutations is None:
            num_permutations = self.num_permutations
            
        batch_size, num_features = x.shape
        scores = []
        
        for _ in range(num_permutations):
            perm = torch.randperm(num_features, device=x.device)
            loss, _, _ = self.forward_single(x, perm=perm, causal=True)
            scores.append(loss)
        
        scores = torch.stack(scores, dim=0)  # (num_perms, B)
        
        if aggregation == 'mean':
            return scores.mean(dim=0)
        elif aggregation == 'max':
            return scores.max(dim=0).values
        elif aggregation == 'median':
            return scores.median(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    @torch.no_grad()
    def compute_anomaly_score_detailed(
        self, 
        x: torch.Tensor,
        num_permutations: Optional[int] = None,
    ) -> dict:

        if num_permutations is None:
            num_permutations = self.num_permutations
            
        B, F = x.shape
        scores = []
        
        for _ in range(num_permutations):
            perm = torch.randperm(F, device=x.device)
            loss, _, _ = self.forward_single(x, perm=perm, causal=True)
            scores.append(loss)
        
        scores = torch.stack(scores, dim=0)  # (num_perms, B)
        
        return {
            'mean': scores.mean(dim=0),
            'max': scores.max(dim=0).values,
            'min': scores.min(dim=0).values,
            'std': scores.std(dim=0),
            'median': scores.median(dim=0).values,
        }


class NEPA_TAD_WithReconstruction(NEPA_TAD):
    """
    NEPA-TAD with optional reconstruction head for hybrid scoring
    """
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        dropout_prob: float = 0.0,
        use_rope: bool = True,
        layer_scale_init: float = 1e-5,
        num_permutations: int = 10,
        recon_weight: float = 0.0,  # Weight for reconstruction loss
    ):
        super().__init__(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout_prob=dropout_prob,
            use_rope=use_rope,
            layer_scale_init=layer_scale_init,
            num_permutations=num_permutations,
        )
        
        self.recon_weight = recon_weight
        
        if recon_weight > 0:
            # Simple reconstruction head
            self.recon_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            print(f"Reconstruction head enabled with weight={recon_weight}")
    
    def forward(
        self, 
        x: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor:
        B, F = x.shape
        
        # Random permutation
        perm = torch.randperm(F, device=x.device)
        
        # Permute input
        x_perm = x[:, perm]
        
        loss, z, z_hat = self.forward_single(x, perm=perm, causal=True)
        
        # Add reconstruction loss if enabled
        if self.recon_weight > 0:
            x_recon = self.recon_head(z_hat).squeeze(-1)  # (B, F)
            recon_loss = F.mse_loss(x_recon, x_perm, reduction='none').mean(dim=-1)
            loss = loss + self.recon_weight * recon_loss
        
        if return_details:
            return loss, z, z_hat, perm
        return loss
