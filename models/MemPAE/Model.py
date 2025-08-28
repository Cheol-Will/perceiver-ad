import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random


class MemoryUnit(nn.Module):
    def __init__(
        self,
        num_memories: int,
        hidden_dim: int,
        sim_type: str,    # "cos" or "l2"
        temperature: float = 1.0,
    ):
        super().__init__()
        assert sim_type.lower() in ['cos', 'l2']
        print(f"Init MemoryUnit of shape {num_memories, hidden_dim} with simtype={sim_type} and t={temperature}")
        self.memories = nn.Parameter(torch.empty(num_memories, hidden_dim))
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.sim_type = sim_type.lower()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.memories)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, D), memories: (N, D)
        if self.sim_type == "cos":
            x_norm = F.normalize(x, dim=-1)                  # (B, K, D)
            mem_norm = F.normalize(self.memories, dim=-1)    # (N, D)
            logits = x_norm @ mem_norm.t()                   # (B, K, N)

        elif self.sim_type == "l2":
            x_sq = (x ** 2).sum(dim=2, keepdim=True)             # (B, K, 1)
            m_sq = (self.memories ** 2).sum(dim=1, keepdim=True).t()  # (1, N)
            dist_sq = x_sq + m_sq - 2 * (x @ self.memories.t())  # (B, K, N)
            dist_sq = dist_sq.clamp_min(0.) 
            logits = -dist_sq                                    

        else:
            raise ValueError(f"sim_type must be 'cos' or 'l2', got {self.sim_type}")

        logits = logits / self.temperature
        weight = F.softmax(logits, dim=-1)                       # (B, K, N)
        read = weight @ self.memories                            # (B, K, D)
        return read  # (B, K, D)

class MLP(nn.Module):
    """A dense module following attention in Transformer block."""
    def __init__(
        self,
        dim: int,
        dim_multiplier: float = 2.0,
        drop: float = 0.0,
    ):
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
    """
        Multihead attention module
    """
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

    def reset_parameters(self,):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x_q, x_k, x_v, return_weight = False):
        """
            x_q: (B, S, D)
            x_k: (B, F, D)
            x_v: (B, F, D)
        """
        assert x_q.ndim == 3
        q = self.q(x_q) # (B, S, D)
        k = self.k(x_k) # (B, F, D)
        v = self.v(x_v) # (B, F, D)
        B, S, D = q.shape
        _, F, D = k.shape

        q = q.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, S, D_H)
        k = k.reshape(B, F, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, F, D_H)
        v = v.reshape(B, F, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, F, D_H)

        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale 
        attn = torch.einsum('bhsd,bhfd->bhsf', q, k) # (B, H, S, F)
        attn = attn.softmax(dim=-1) # (B, H, S, F)
        attn = self.attn_drop(attn)  
        x = torch.einsum('bhsf,bhfd->bhsd', attn, v)
        x = x.transpose(1, 2).reshape(B, S, -1)

        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_weight:
            return x, attn 
        else:
            return x # (B, S ,D)

class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_prob: float = 0.0,
    ):
        super(CrossAttention, self).__init__()
        self.attention = MultiHeadAttention(
            dim=hidden_dim,
            num_heads=num_heads,
        )     
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout_prob)    

    def forward(self, x_q, x_k, x_v, return_weight = False):
        if return_weight:
            x_attn, attn = self.attention(x_q, x_k, x_v, return_weight)
            x_q = x_q + x_attn
            x_q = x_q + self.mlp(x_q)
            return x_q, attn
        else: 
            x_q = x_q + self.attention(x_q, x_k, x_v, return_weight)
            x_q = x_q + self.mlp(x_q)
            return x_q
    
class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_prob: float = 0.0,
    ):
        super(SelfAttention, self).__init__()

        self.attention = MultiHeadAttention(
            dim=hidden_dim,
            num_heads=num_heads,
        )     
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout_prob)    

    def forward(self, x, return_weight = False):
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
        """
        x: (batch_size, num_features)
        return: (batch_size, num_features, hidden_dim)
        """
        x_embedded = x.unsqueeze(-1) * self.embedding_matrix  # (batch_size, num_features, hidden_dim)
        return x_embedded

class OutputProjection(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(OutputProjection, self).__init__()
        self.embedding_matrix = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding_matrix)
        
    def forward(self, x):
        """
        x: (batch_size, num_features, hidden_dim)
        return: (batch_size, num_features)
        """
        x = x * self.embedding_matrix  # (batch_size, num_features, hidden_dim)
        x = x.sum(dim=-1)  # (batch_size, num_features)
        
        return x


class MemPAE(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_heads: int = 4,
        depth: int = 4, 
        hidden_dim: int = 64,
        mlp_ratio: float = 4,
        dropout_prob: float = 0.0,
        num_latents: int = None,
        num_memories: int = None,
        is_weight_sharing: bool = True,
        temperature: float = 1,
        sim_type: str = 'cos',
    ):
        super(MemPAE, self).__init__()
        assert num_latents is not None
        assert num_memories is not None
        print("Init MemPAE with weight_sharing" if is_weight_sharing else "Init MemPAE without weight sharing")
        
        self.feature_tokenizer = FeatureTokenizer(num_features, hidden_dim) # only numerical inputs
        self.memory = MemoryUnit(num_memories, hidden_dim, sim_type, temperature)
        
        if is_weight_sharing:
            self.block = SelfAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob)
        else:
            self.block = nn.ModuleList([
                SelfAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob)
                for _ in range(depth)
            ])
        
        self.encoder = CrossAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob)
        self.decoder = CrossAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob)
        self.proj = OutputProjection(num_features, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        self.decoder_query = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        self.latents_query = nn.Parameter(torch.empty(1, num_latents, hidden_dim))
        
        self.num_features = num_features
        self.is_weight_sharing = is_weight_sharing
        self.depth = depth
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.decoder_query, std=0.02)
        nn.init.trunc_normal_(self.latents_query, std=0.02)
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def forward(self, x, return_weight = False):
        batch_size, num_features = x.shape # (B, F)

        # feature tokenizer
        feature_embedding = self.feature_tokenizer(x) # (B, F, D)
        feature_embedding = feature_embedding + self.pos_encoding

        # encoder 
        latents_query = self.latents_query.expand(batch_size, -1, -1) # (B, N, D)
        latents = self.encoder(latents_query, feature_embedding, feature_embedding) 

        # self attention
        if self.is_weight_sharing:
            for _ in range(self.depth):
                latents = self.block(latents)
        else:
            for block in self.block:
                latents = block(latents)

        # memory addressing
        latents = self.memory(latents) 

        # decoder
        decoder_query = self.decoder_query.expand(batch_size, -1, -1) # (B, F, D)
        output = self.decoder(decoder_query, latents, latents)
        x_hat = self.proj(output)

        loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=1) # keep batch dim

        return loss