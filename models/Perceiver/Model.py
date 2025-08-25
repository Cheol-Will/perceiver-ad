import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import itertools

def comb(n, r):
    if r < 0 or r > n:
        return 0
    r = min(r, n - r) # symmetric
    numer = 1
    denom = 1
    for i in range(1, r+1):
        numer *= n - (r - i)
        denom *= i
    return numer // denom


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
    def __init__(
        self, 
        num_features,
        input_dim,
        hidden_dim,
    ):
        super(FeatureTokenizer, self).__init__()
        self.embeddings = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_features)])
   
    def forward(self, x, col_idx):
        """
        x: (batch_size, num_features)
        col_idx: feature index
        return: (batch_size, col_idx_len, hidden_dim)
        """
        outs = []
        for i in col_idx:
            col = x[:, i].unsqueeze(-1)          # (batch_size, 1)
            out = self.embeddings[i](col)        # (batch_size, hidden_dim)
            outs.append(out.unsqueeze(1))        # (batch_size, 1, hidden_dim)

        return torch.cat(outs, dim=1)  # (batch_size, len(col_idx), hidden_dim)


class Perceiver(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_heads: int = 4,
        depth: int = 4, 
        hidden_dim: int = 64,
        mlp_ratio: float = 4,
        dropout_prob: float = 0.0,
        drop_col_prob: float = 0.5,
        max_repeat: int = 100,
    ):
        assert 0.0 < drop_col_prob < 1.0 
        super(Perceiver, self).__init__()
        self.feature_tokenizer = FeatureTokenizer(num_features, 1, hidden_dim) # only numerical inputs
        self.block = nn.ModuleList([
            SelfAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob)
            for _ in range(depth)
        ])
            
        self.decoder = CrossAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob)
        self.proj = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_features)])
        self.pos_encoding = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        self.decoder_query = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        
        self.num_features = num_features
        self.drop_col_prob = drop_col_prob
        self.max_repeat = max_repeat
        self.test_drop_col = self.generate_test_drop_col()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.decoder_query, std=0.02)
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def generate_test_drop_col(self):
        n = self.num_features
        k = max(1, math.floor(n * self.drop_col_prob))
        # max_num_combinations = math.comb(n, k)
        max_num_combinations = comb(n, k)

        # if max_repeat > nCk
        if self.max_repeat >= max_num_combinations:
            return [list(c) for c in itertools.combinations(range(n), k)]
        
        combs = []
        visited = set()
        while len(combs) < self.max_repeat:
            c = tuple(sorted(random.sample(range(n), k)))
            if c not in combs:
                visited.add(c)
                combs.append(list(c))
    
        return combs    

    def generate_random_drop_col(self):
        assert self.training
        cols = list(range(self.num_features))
        drop_col_indicies = [i for i in cols if torch.rand(1).item() < self.drop_col_prob]
        if len(drop_col_indicies) == 0:
            # Drop at least one column
            drop_col_indicies.append(torch.randint(0, self.num_features, (1,)).item())
        elif len(drop_col_indicies) == self.num_features:
            # Do not drop every columns.
            alive_idx = torch.randint(0, self.num_features, (1,)).item()
            drop_col_indicies.remove(alive_idx)
        return [drop_col_indicies]

    def forward(self, x, return_weight = False):
        batch_size, num_features = x.shape # (B, F)

        if self.training:
            target_col_idx_list = self.generate_random_drop_col()
        else:
            target_col_idx_list = self.test_drop_col
        
        batch_losses = torch.zeros(batch_size, device=x.device, dtype=x.dtype) # 
        for target_col_idx in target_col_idx_list:
            # Indexing input and target
            input_col_idx = [i for i in range(num_features) if i not in target_col_idx]
            target = x[:, target_col_idx] # (B, F-F')

            # Embed input and add pos encoding
            encoding = self.feature_tokenizer(x, input_col_idx) # (B, F', D)
            encoding = encoding + self.pos_encoding[:, input_col_idx, :]

           # transformer block
            if return_weight:
                attns = []
                for block in self.block:
                    encoding, attn = block(encoding, return_weight=True)
                    attns.append(attn)
            else:
                for block in self.block: 
                    encoding = block(encoding)

            # decoding 
            decoder_query = self.decoder_query[:, target_col_idx, :].expand(batch_size, -1, -1)
            
            if return_weight:
                decoding, atnn = self.decoder(decoder_query, encoding, encoding, return_weight=True)
                attns.append(atnn)
            else:
                decoding = self.decoder(decoder_query, encoding, encoding)

            pred = []
            for i, col_idx in enumerate(target_col_idx):
                pred.append(self.proj[col_idx](decoding[:, i, :]))
            pred = torch.cat(pred, dim=1) # (B, F-F')

            # output error
            loss = F.mse_loss(pred, target, reduction='none').mean(dim=1) # keep batch dim
            batch_losses = batch_losses + loss # (B)

        batch_losses = batch_losses / len(target_col_idx_list)
        
        if self.training and return_weight:
            return batch_losses, attns, input_col_idx, target_col_idx 
        else:
            return batch_losses