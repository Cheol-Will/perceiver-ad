import torch
import torch.nn as nn
import torch.nn.functional as F


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
        use_flash_attn: bool = False,  # 추가
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attn = use_flash_attn  # 추가
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias) 
        self.k = nn.Linear(dim, dim, bias=qkv_bias) 
        self.v = nn.Linear(dim, dim, bias=qkv_bias) 
        
        self.q_norm = getattr(nn, normalization)(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = getattr(nn, normalization)(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_p = attn_drop  # Flash Attention용 dropout 확률 저장
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
                    
    def forward(self, x_q, x_k, x_v, return_weight=False):
        assert x_q.ndim == 3
        q = self.q(x_q)
        k = self.k(x_k)
        v = self.v(x_v)
        B, S, D = q.shape
        _, L, _ = k.shape  # F → L로 변경

        q = q.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_flash_attn and not return_weight:
            # Flash Attention
            dropout_p = self.attn_drop_p if self.training else 0.0
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
            )
            attn = None
        else:
            q = q * self.scale
            attn = torch.einsum('bhsd,bhld->bhsl', q, k)  # f → l
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = torch.einsum('bhsl,bhld->bhsd', attn, v)  # f → l

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
        use_flash_attn: bool = False,  
    ):
        super(SelfAttention, self).__init__()

        self.attention = MultiHeadAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            use_flash_attn=use_flash_attn,  
        )     
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


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_prob: float = 0.0,
        use_flash_attn: bool = False,  
    ):
        super(CrossAttention, self).__init__()
        self.attention = MultiHeadAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            use_flash_attn=use_flash_attn,  
        )     
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout_prob)    

    def forward(self, x_q, x_k, x_v, return_weight=False):
        if return_weight:
            x_attn, attn = self.attention(x_q, x_k, x_v, return_weight)
            x_q = x_q + x_attn
            x_q = x_q + self.mlp(x_q)
            return x_q, attn
        else: 
            x_q = x_q + self.attention(x_q, x_k, x_v, return_weight)
            x_q = x_q + self.mlp(x_q)
            return x_q


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


class BaseEncoder(nn.Module):
    def __init__(self, 
        num_features,
        hidden_dim,
        depth,
        num_heads=4,
        mlp_ratio=4.0,
        dropout_prob=0.0,
        use_flash_attn=False,
    ):
        super(BaseEncoder, self).__init__()
        self.feature_tokenizer = FeatureTokenizer(num_features, hidden_dim)
        self.encoder = nn.Sequential(*[
            SelfAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob, use_flash_attn) 
            for _ in range(depth)
        ])
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.pos_encoding = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]
        embedding = self.feature_tokenizer(x) + self.pos_encoding.expand(batch_size, -1, -1)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        embedding = torch.cat([cls_token, embedding], dim=1) 
        embedding = self.encoder(embedding)
        cls_token = embedding[:, 0, :]  # (B, D)
        return cls_token


class BaseDecoder(nn.Module):
    def __init__(self, 
        num_features,
        hidden_dim,
        num_heads=8,
        mlp_ratio=4.0,
        dropout_prob=0.0,
        use_flash_attn=False,
    ):
        super(BaseDecoder, self).__init__()
        self.decoder = SelfAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob, use_flash_attn)
        self.output_projection = OutputProjection(num_features, hidden_dim)

    def forward(self, cls_token, pos_encoding):
        """
        cls_token: (B, D)
        pos_encoding: (1, F, D)
        """
        batch_size = cls_token.shape[0]
        cls_token = cls_token.unsqueeze(1)  # (B, 1, D)
        embedding_dec = torch.cat([cls_token, pos_encoding.expand(batch_size, -1, -1)], dim=1)
        embedding_dec = self.decoder(embedding_dec)
        embedding_dec = embedding_dec[:, 1:, :]  # Remove cls token
        x_hat = self.output_projection(embedding_dec)
        return x_hat