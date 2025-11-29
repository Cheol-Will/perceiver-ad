import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

class EntropyLoss(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        if x.dim() == 3:
            # x: (B, N, M)
            b = x * torch.log(x + self.eps)
            b = -1.0 * b.sum(dim=-1) # (B, N)
            b = b.mean(dim=-1) # (B, )
            return b
        elif x.dim() == 4:
            # x: (B, H, N, M)
            b = x * torch.log(x + self.eps)
            b = -1.0 * b.sum(dim=-1) # (B, H, N)
            b = b.mean(dim=-1).mean(dim=-1) # (B, )
            return b
        
class MemoryUnit(nn.Module):
    def __init__(
        self,
        num_memories: int,
        hidden_dim: int,
        sim_type: str,    
        temperature: float = 1.0,
        shrink_thres: float = 0,
        top_k: int = None,
        num_heads: int = None,
    ):
        super().__init__()
        assert sim_type.lower() in ['cos', 'l2', 'attn', 'cross_attn']
        if sim_type.lower() == 'attn':
            assert num_heads is not None
        print(f"Init MemoryUnit of shape {num_memories, hidden_dim} \
              with simtype={sim_type} and t={temperature} thres={shrink_thres}")
        print(f"top_k={top_k}")
        self.memories = nn.Parameter(torch.empty(num_memories, hidden_dim)) # (M, D)
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.shrink_thres = shrink_thres
        self.scale = self.hidden_dim**(-0.5)
        self.sim_type = sim_type.lower()
        self.top_k = top_k

        if self.sim_type == 'cross_attn':
            self.cross_attn = MultiHeadAttention(dim=hidden_dim, num_heads=num_heads)
        elif self.sim_type == 'attn':
            self.q_norm = nn.LayerNorm(hidden_dim)
            self.k_norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.memories)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D), memories: (M, D)
        if self.sim_type == "cos":
            x_norm = F.normalize(x, dim=-1) # (B, N, D)                 
            mem_norm = F.normalize(self.memories, dim=-1) # (M, D)
            logits = x_norm @ mem_norm.t() # (B, N, M)

        elif self.sim_type == "l2":
            x_sq = (x ** 2).sum(dim=2, keepdim=True)             
            m_sq = (self.memories ** 2).sum(dim=1, keepdim=True).t()  
            dist_sq = x_sq + m_sq - 2 * (x @ self.memories.t())  
            dist_sq = dist_sq.clamp_min(0.) # to avoid sqrt of negative value 
            logits = -dist_sq                                    

        elif self.sim_type == 'attn':            
            q = self.q_norm(x) 
            k = self.k_norm(self.memories)
            q = q * self.scale
            logits = q @ k.t()

        elif self.sim_type == 'cross_attn':
            batch_size = x.shape[0]
            memories = self.memories.unsqueeze(0).expand(batch_size, -1, -1) # (B, M, D)
            read, weight = self.cross_attn(x, memories, memories, return_weight=True)
            return read, weight
        
        else:
            raise ValueError(f"Unknown sim_type is given: {self.sim_type}")

        if self.top_k is not None:
            top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
            top_k_weights = F.softmax(top_k_logits / self.temperature, dim=-1) # (B, N, K)
            weight = torch.zeros_like(logits) # (B, N, M)
            weight = weight.scatter(-1, top_k_indices, top_k_weights)            
        else:
            logits = logits / self.temperature
            weight = F.softmax(logits, dim=-1)
            
            if self.shrink_thres > 0:
                weight = hard_shrink_relu(weight, lambd=self.shrink_thres)
                weight = F.normalize(weight, p=1, dim=-1)

        read = weight @ self.memories # (B, N, M) @ (M, D) = (B, N, D)                           
        return read, weight  

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
        

class LowRankAttention(nn.Module):
    def __init__(
        self,
        rank: int,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_prob: float = 0.0,
    ):
        super(LowRankAttention, self).__init__()
        
        self.inducing_points = nn.Parameter(torch.empty(1, rank, hidden_dim))
        self.low_rank_attention = CrossAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob)
        self.high_rank_attention = CrossAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.inducing_points)  


    def forward(self, x):
        B = x.shape[0]
        inducing = self.inducing_points.expand(B, -1, -1)  # (B, rank, D)

        x_low_rank = self.low_rank_attention(inducing, x, x)
        x = self.high_rank_attention(x, x_low_rank, x_low_rank)
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

class MLPMixerEncoder(nn.Module):
    def __init__(
        self, 
        hidden_dim,
        num_features,
        num_latents,
        dropout_prob,
    ):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_prob)
        self.lin2 = nn.Linear(num_features, num_latents)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        assert x.ndim == 3 # (B, N, D)
        x = self.drop1(self.act1(self.lin1(x))) # B, N, D
        x = x.permute(0, 2, 1) # B, D, N
        x = self.drop2(self.act2(self.lin2(x))) # B, D, N
        x = x.permute(0, 2, 1) # B, N, D

        return x

class MLPMixerDecoder(nn.Module):
    def __init__(
        self, 
        hidden_dim,
        num_latents,
        num_features,
        dropout_prob,
    ):
        super().__init__()
        self.lin1 = nn.Linear(num_latents, hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_prob)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_prob)
        self.lin3 = nn.Linear(hidden_dim, num_features)
        
    def forward(self, x):
        assert x.ndim == 3
        # print(f"in dim: {x.shape}")
        x = x.permute(0, 2, 1) # B, D, N
        # print(f"in dim: {x.shape}")
        x = self.drop1(self.act1(self.lin1(x))) # B, D, H
        x = self.drop2(self.act2(self.lin2(x))) # B, D, H
        x = self.lin3(x) # B, D, F
        x = x.permute(0, 2, 1) # B, F, D

        return x

class MLPEncoder(nn.Module):
    def __init__(
        self, 
        hidden_dim: int,
        num_features,
        dropout_prob: float,
    ):
        super().__init__()
        self.lin1 = nn.Linear(num_features, hidden_dim)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_prob)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_prob)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1) # (B, 1, F)
        x = self.drop1(self.act1(self.lin1(x))) # B, 1, D
        x = self.drop2(self.act2(self.lin2(x))) # B, 1, D
        x = self.lin3(x) # B, 1, D

        return x


class MLPDecoder(nn.Module):
    def __init__(
        self, 
        hidden_dim: int,
        num_latents: int,
        num_features: int,
        dropout_prob: float,
    ):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim*num_latents, hidden_dim*num_latents)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_prob)
        self.lin2 = nn.Linear(hidden_dim*num_latents, hidden_dim*num_latents)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_prob)
        self.lin3 = nn.Linear(hidden_dim*num_latents, num_features)
        
    def forward(self, x):
        # make output (B, F)
        assert x.ndim == 3

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.drop1(self.act1(self.lin1(x))) # B, ND
        x = self.drop2(self.act2(self.lin2(x))) # B, ND
        x = self.lin3(x) # B, F

        return x


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


class MemSet(nn.Module):
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
        is_weight_sharing: bool = False,
        temperature: float = 1,
        sim_type: str = 'cos',
        shrink_thred: float = 0.0,
        use_mask_token: bool = False,
        use_pos_enc_as_query: bool = False,
        latent_loss_weight: float = None,
        use_latent_loss_as_score: bool = False,
        entropy_loss_weight: float = None,
        use_entropy_loss_as_score: bool = False,
        top_k: int = None,
        is_recurrent: bool = False,
        mlp_encoder: bool = False, # make one token
        mlp_mixer_encoder: bool = False, # make one token
        mlp_decoder: bool = False, # flatten latent_hat and apply mlp
        mlp_mixer_decoder: bool = False, # 
        global_decoder_query: bool = False,
        not_use_memory: bool = False,
        not_use_decoder: bool = False,
        use_pos_embedding: bool = False,
    ):
        super(MemSet, self).__init__()
        assert num_latents is not None
        assert num_memories is not None
        if use_latent_loss_as_score:
            assert latent_loss_weight is not None
        if use_entropy_loss_as_score:
            assert entropy_loss_weight is not None
        if not use_pos_enc_as_query:
            assert not use_mask_token
        if top_k is not None:
            top_k = min(top_k, num_memories)

        self.pos_embedding = nn.Parameter(torch.empty(1, num_features, hidden_dim)) if use_pos_embedding else None
        self.feature_tokenizer = FeatureTokenizer(num_features, hidden_dim) # only numerical inputs
        self.memory = MemoryUnit(num_memories, hidden_dim, sim_type, temperature, shrink_thred, top_k, num_heads)
        print("Do not use memory module" if not_use_memory else "Use memory module")
                
        self.blocks = nn.Sequential(*[
            LowRankAttention(
                rank=num_latents,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout_prob=dropout_prob,
                ) for _ in range(depth)]
        )
        self.proj = OutputProjection(num_features, hidden_dim)
        self.not_use_memory = not_use_memory
        self.reset_parameters()

    def reset_parameters(self):
        if self.pos_embedding is not None:
            nn.init.xavier_uniform_(self.pos_embedding)

    def forward(
        self, 
        x, 
    ):
        feature_embedding = self.feature_tokenizer(x) # (B, F, D)
        feature_embedding = self.blocks(feature_embedding)
        feature_embedding_hat, _ = self.memory(feature_embedding)
        x_hat = self.proj(feature_embedding_hat)
        loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=1) # keep batch dim
        return loss