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
        x = x.permute(0, 2, 1) # B, D, N
        x = self.drop1(self.act1(self.lin1(x))) # B, D, H
        x = self.drop2(self.act2(self.lin2(x))) # B, D, H
        x = self.lin3(x) # B, D, F
        x = x.permute(0, 2, 1) # B, F, D

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
        mlp_mixer_decoder: bool = False,
    ):
        super(MemPAE, self).__init__()
        assert num_latents is not None
        assert num_memories is not None
        if use_latent_loss_as_score:
            assert latent_loss_weight is not None
        if use_entropy_loss_as_score:
            assert entropy_loss_weight is not None
        if not use_pos_enc_as_query:
            assert not use_mask_token

        print("Init MemPAE with weight_sharing" if is_weight_sharing else "Init MemPAE without weight sharing")
        print(f"latent_loss_weight={latent_loss_weight} and entropy_loss_weight={entropy_loss_weight}")

        self.feature_tokenizer = FeatureTokenizer(num_features, hidden_dim) # only numerical inputs
        self.memory = MemoryUnit(num_memories, hidden_dim, sim_type, temperature, shrink_thred, top_k, num_heads)
        
        if is_weight_sharing:
            self.block = SelfAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob)
        else:
            self.block = nn.ModuleList([
                SelfAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob)
                for _ in range(depth)
            ])
        
        self.encoder = CrossAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob)
        if mlp_mixer_decoder: 
            # input: (B, N, D)
            # output: (B, F)
            print("Init MLPMixerDecoder.")
            self.decoder = MLPMixerDecoder(  
                hidden_dim,
                num_latents,
                num_features,
                dropout_prob
            )
        else:
            self.decoder = CrossAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob)

        self.proj = OutputProjection(num_features, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        self.latents_query = nn.Parameter(torch.empty(1, num_latents, hidden_dim))

        if use_pos_enc_as_query:
            if use_mask_token:
                print(f"Init decoder query of shape {(1, 1, hidden_dim)} and use decoder query + pos_encoding as query token.")
                self.decoder_query = nn.Parameter(torch.empty(1, 1, hidden_dim)) # 1 x d
            else:
                print(f"Do not init decoder query but use positional encoding as decoder query")
                self.decoder_query = None # 
        else:
            print(f"Init decoder query of shape {(1, num_features, hidden_dim)}")
            self.decoder_query = nn.Parameter(torch.empty(1, num_features, hidden_dim)) # f x d

        self.entropy_loss_fn = EntropyLoss()

        self.num_features = num_features
        self.is_weight_sharing = is_weight_sharing
        self.use_pos_enc_as_query = use_pos_enc_as_query
        self.use_mask_token = use_mask_token
        self.depth = depth
        self.latent_loss_weight = latent_loss_weight
        self.use_latent_loss_as_score = use_latent_loss_as_score
        self.entropy_loss_weight = entropy_loss_weight
        self.use_entropy_loss_as_score = use_entropy_loss_as_score
        self.is_recurrent = is_recurrent
        self.mlp_mixer_decoder = mlp_mixer_decoder
        self.reset_parameters()

    def reset_parameters(self):
        if self.decoder_query is not None:
            nn.init.trunc_normal_(self.decoder_query, std=0.02)
        nn.init.trunc_normal_(self.latents_query, std=0.02)
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def forward(
        self, 
        x, 
        return_for_analysis: bool = False,
        return_attn_weight: bool = False, 
        return_pred: bool = False, 
        return_memory_weight: bool = False,
        return_latents: bool = False,
        return_pred_all: bool = False,
    ):
    
        batch_size, num_features = x.shape # (B, F)

        # feature tokenizer
        feature_embedding = self.feature_tokenizer(x) # (B, F, D)
        feature_embedding = feature_embedding + self.pos_encoding

        # encoder 
        latents_query = self.latents_query.expand(batch_size, -1, -1) # (B, N, D)
        latents, attn_weight_enc = self.encoder(latents_query, feature_embedding, feature_embedding, return_weight=True) 

        # self attention
        attn_weight_self_list = []
        if self.is_weight_sharing:
            for _ in range(self.depth):
                latents, attn_weight_self = self.block(latents, return_weight=True)
                attn_weight_self_list.append(attn_weight_self)
                if self.is_recurrent:
                    latents, _ = self.memory(latents)
        else:
            for block in self.block:
                latents ,attn_weight_self = block(latents, return_weight=True)
                attn_weight_self_list.append(attn_weight_self)

        if self.is_recurrent:
            latents_hat = latents # addressing is already performed
        else:    
            # memory addressing
            latents_hat, memory_weight = self.memory(latents) # (B, N, D), (B, N, M) 

        # decoder
        if self.use_pos_enc_as_query:
            if self.use_mask_token:
                decoder_query = self.decoder_query.expand(batch_size, num_features, -1) # (1, 1, D) -> (B, F, D)
                decoder_query = decoder_query + self.pos_encoding
            else:
                decoder_query = self.pos_encoding.expand(batch_size, -1, -1)
        else:
            decoder_query = self.decoder_query.expand(batch_size, -1, -1) # (B, F, D)

        if self.mlp_mixer_decoder:
            attn_weight_dec = None
            output = self.decoder(latents_hat)
        else:
            output, attn_weight_dec = self.decoder(decoder_query, latents_hat, latents_hat, return_weight=True)
        
        x_hat = self.proj(output)
        loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=1) # keep batch dim

        # latent loss
        if self.latent_loss_weight is not None:
            if self.training:
                latent_loss = F.mse_loss(latents_hat, latents, reduction='none').mean(dim=[1,2]) # (B, )
                loss = loss + self.latent_loss_weight * latent_loss 
            else:
                if self.use_latent_loss_as_score:   
                    latent_loss = F.mse_loss(latents_hat, latents, reduction='none').mean(dim=[1,2]) # (B, )
                    loss = loss + self.latent_loss_weight * latent_loss 

        # memory addressing entropy loss
        if self.entropy_loss_weight is not None:              
            if self.training:      
                entropy_loss = self.entropy_loss_fn(memory_weight)
                loss = loss + self.entropy_loss_weight * entropy_loss
            else:
                if self.use_entropy_loss_as_score:
                    entropy_loss = self.entropy_loss_fn(memory_weight)
                    loss = loss + self.entropy_loss_weight * entropy_loss

        if return_for_analysis:
            return loss, x, x_hat, latents, latents_hat, memory_weight, attn_weight_enc, attn_weight_self_list, attn_weight_dec

        # for analysis
        if return_latents:
            return loss, latents, latents_hat

        if return_attn_weight:
            return loss, attn_weight_enc, attn_weight_self_list, attn_weight_dec

        if return_pred_all:
            output_origin, _ = self.decoder(decoder_query, latents, latents, return_weight=True)
            x_hat_origin = self.proj(output_origin)
            x_hat_memory = x_hat
            
            return x, x_hat_origin, x_hat_memory

        if return_pred:
            if return_memory_weight:
                return loss, x, x_hat, memory_weight
            else:
                return loss, x, x_hat
        else:
            if return_memory_weight:
                return loss, memory_weight
            else:
                return loss