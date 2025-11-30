import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from itertools import combinations

class MLP(nn.Module):
    def __init__(self, dim: int, dim_multiplier: float = 2.0, drop: float = 0.0):
        super(MLP, self).__init__()
        hidden_dim = int(dim * dim_multiplier)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)
        self.reset_parameters()

    def reset_parameters(self):
       for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.norm(x)
        x = self.drop2(self.fc2(x))
        return x        

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        scale_norm: bool = True,
        proj_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        normalization: str = 'LayerNorm',
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias) 
        self.k = nn.Linear(dim, dim, bias=qkv_bias) 
        self.v = nn.Linear(dim, dim, bias=qkv_bias) 
        
        self.q_norm = getattr(nn, normalization)(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = getattr(nn, normalization)(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = getattr(nn, normalization)(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x_q, x_k, x_v, return_weight=False):
        assert x_q.ndim == 3
        q = self.q(x_q)
        k = self.k(x_k)
        v = self.v(x_v)
        B, S, D = q.shape
        _, F, D = k.shape

        q = q.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, F, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, F, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale 
        attn = torch.einsum('bhsd,bhfd->bhsf', q, k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  
        x = torch.einsum('bhsf,bhfd->bhsd', attn, v)
        x = x.transpose(1, 2).reshape(B, S, -1)

        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_weight:
            return x, attn 
        else:
            return x
    
class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_prob: float = 0.0,
    ):
        super(SelfAttention, self).__init__()
        self.attention = MultiHeadAttention(dim=hidden_dim, num_heads=num_heads)     
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout_prob)    

    def forward(self, x, return_weight=False):
        if return_weight:
            x_attn, attn = self.attention(x, x, x, return_weight)
            x = x + x_attn
            x = x + self.mlp(x)
            return x, attn
        else: 
            x = x + self.attention(x, x, x, return_weight)
            x = x + self.mlp(x)
            return x    

class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(FeatureTokenizer, self).__init__()
        self.embedding_matrix = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding_matrix)
        
    def forward(self, x):
        """x: (N, D) -> (N, D, E)"""
        x_embedded = x.unsqueeze(-1) * self.embedding_matrix
        return x_embedded

class OutputProjection(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(OutputProjection, self).__init__()
        self.embedding_matrix = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding_matrix)
        
    def forward(self, x):
        """x: (N, D, E) -> (N, D)"""
        x = x * self.embedding_matrix
        x = x.sum(dim=-1)
        return x

class NPTAD(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_heads: int = 4,
        depth: int = 4, 
        hidden_dim: int = 64,
        mlp_ratio: float = 4,
        dropout_prob: float = 0.0,
        train_mask_prob: float = 0.15,
        test_mask_prob: float = 0.15,
        num_reconstructions: int = 15,
        max_masked_features: int = None,
    ):
        super(NPTAD, self).__init__()

        self.feature_tokenizer = FeatureTokenizer(num_features, hidden_dim)
        self.proj = OutputProjection(num_features, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        
        # ABD: row attention over samples
        self.row_attention = nn.ModuleList([
            SelfAttention(hidden_dim * num_features, num_heads, mlp_ratio, dropout_prob)
            for _ in range(depth)
        ])
        
        # ABA: column attention over features
        self.col_attention = nn.ModuleList([
            SelfAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob)
            for _ in range(depth)
        ])

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.train_mask_prob = train_mask_prob
        self.test_mask_prob = test_mask_prob
        self.num_reconstructions = num_reconstructions
        
        # Generate mask bank for evaluation
        if max_masked_features is None:
            max_masked_features = max(1, int(num_features * test_mask_prob))
        self.max_masked_features = max_masked_features
        # self.mask_bank = self._generate_mask_bank(num_features, max_masked_features)
        self.mask_bank = None
        
        # print(f"Mask bank: {len(self.mask_bank)} masks, max {max_masked_features} features")
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def _generate_mask_bank(self, num_features, max_masked):
        """Generate all combinations of masked features"""
        mask_bank = []
        for num_masked in range(1, max_masked + 1):
            for combo in combinations(range(num_features), num_masked):
                mask = torch.zeros(num_features)
                mask[list(combo)] = 1
                mask_bank.append(mask)
        return torch.stack(mask_bank)

    def _create_mask(self, x, mask_prob):
        """Random masking for training"""
        if mask_prob == 0:
            return torch.zeros_like(x)
        mask = torch.bernoulli(torch.full_like(x, mask_prob))
        return mask

    def forward(self, X, mode='train', query_indices=None, mask_idx=None):
        """
        Args:
            X: (N, D) combined support + query samples
            mode: 'train' or 'inference'
            query_indices: indices of query samples
            mask_idx: specific mask index for evaluation
        """
        N, D = X.shape
        device = X.device
        
        # Determine masking strategy
        if mode == 'train':
            M = self._create_mask(X, self.train_mask_prob)
        else:
            if mask_idx is not None:
                M = self.mask_bank[mask_idx].unsqueeze(0).expand(N, -1).to(device)
            else:
                M = self._create_mask(X, self.test_mask_prob)
        
        # Apply masking
        X_masked = X * (1 - M)
        
        # Feature tokenization and positional encoding
        H = self.feature_tokenizer(X_masked)
        H = H + self.pos_encoding
        
        # Alternating ABD and ABA attention
        for row_attn, col_attn in zip(self.row_attention, self.col_attention):
            # ABD: attention between datapoints
            B, D_feat, E = H.shape
            H_flat = H.reshape(B, -1).unsqueeze(0)
            H_flat = row_attn(H_flat).squeeze(0)
            H = H_flat.reshape(B, D_feat, E)
            
            # ABA: attention between attributes
            H = col_attn(H)
        
        # Reconstruction
        X_hat = self.proj(H)
        
        # Compute loss only at masked positions
        recon_loss = F.mse_loss(X_hat, X, reduction='none')
        masked_loss = recon_loss * M
        sample_loss = masked_loss.sum(dim=1)
        num_masked = M.sum(dim=1).clamp(min=1)
        sample_loss = sample_loss / num_masked
        
        # Extract query samples
        if query_indices is not None:
            query_loss = sample_loss[query_indices]
        else:
            query_loss = sample_loss
        
        if mode == 'train':
            return query_loss.mean()
        else:
            return query_loss