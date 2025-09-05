import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

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


class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 hidden_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        print(f"Init VQ layer with n={num_embeddings}, d={hidden_dim}, beta={beta}")
        self.K = num_embeddings
        self.D = hidden_dim
        self.beta = beta
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents = latents.contiguous()
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BN x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BN x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BN, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BN x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BN, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x N x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents, reduction='none').mean(dim=[1,2]) # keep batch dim
        embedding_loss = F.mse_loss(quantized_latents, latents.detach(), reduction='none').mean(dim=[1,2])

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach() # pass gradient to encoder output

        return quantized_latents.contiguous(), vq_loss  # [B x N x D], [B]


class PVQVAE(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_embeddings: int,
        num_heads: int = 4,
        depth: int = 4, 
        hidden_dim: int = 64,
        mlp_ratio: float = 4,
        dropout_prob: float = 0.0,
        num_latents: int = None,
        is_weight_sharing: bool = False,
        use_pos_enc_as_query: bool = False,
        beta: float = 0.25,
    ):
        super().__init__()
        assert num_latents is not None
        print(f"Init PVQVAE with beta={beta}")
        self.feature_tokenizer = FeatureTokenizer(num_features, hidden_dim) # only numerical inputs
        
        if is_weight_sharing:
            print("Init with weight sharing" )
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
        self.latents_query = nn.Parameter(torch.empty(1, num_latents, hidden_dim))
        self.vq_layer = VectorQuantizer(num_embeddings, hidden_dim, beta)

        if use_pos_enc_as_query: 
            print(f"Init decoder query of shape {(1, 1, hidden_dim)}")
            self.decoder_query = nn.Parameter(torch.empty(1, 1, hidden_dim)) # 1 x d
        else:
            self.decoder_query = nn.Parameter(torch.empty(1, num_features, hidden_dim)) # f x d

        self.num_features = num_features
        self.num_latents = num_latents
        self.is_weight_sharing = is_weight_sharing
        self.use_pos_enc_as_query = use_pos_enc_as_query
        self.depth = depth
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.decoder_query, std=0.02)
        nn.init.trunc_normal_(self.latents_query, std=0.02)
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def _encode(self, x):
        batch_size, num_features = x.shape # (B, F)

        # feature tokenizer
        feature_embedding = self.feature_tokenizer(x) # (B, F, D)
        feature_embedding = feature_embedding + self.pos_encoding

        # encoder 
        latents_query = self.latents_query.expand(batch_size, -1, -1) # (B, N, D)
        latents, encoder_attn = self.encoder(latents_query, feature_embedding, feature_embedding, return_weight=True) 

        if self.is_weight_sharing:
            for _ in range(self.depth):
                latents, self_attn = self.block(latents, return_weight=True)
        else:
            for block in self.block:
                latents = block(latents)

        return latents # (B, N, D)

    def _decode(self, latents):
        batch_size = latents.shape[0]
        num_features = self.num_features
        if self.use_pos_enc_as_query:
            decoder_query = self.decoder_query.expand(batch_size, num_features, -1) # (B, F, D)
            decoder_query = decoder_query + self.pos_encoding
        else:
            decoder_query = self.decoder_query.expand(batch_size, -1, -1) # (B, F, D)
        output, decoder_attn = self.decoder(decoder_query, latents, latents, return_weight=True)
        x_hat = self.proj(output)
        return x_hat

    def forward(self, x):
        # decoder
        batch_size = x.shape[0]
        latents = self._encode(x)
        latents, vq_loss = self.vq_layer(latents) # [B, N, D], [B]
        x_hat = self._decode(latents)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=1) # keep batch dim

        return reconstruction_loss, vq_loss