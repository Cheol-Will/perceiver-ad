import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import FeatureTokenizer, SelfAttention, CrossAttention, MultiHeadAttention, OutputProjection

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output
        
def compute_entropy_loss(weight):
    """
    Entropy loss so that memory weight becomes uniform.
    """
    eps = 1e-8
    weight = weight + eps
    
    entropy = -(weight * torch.log(weight)).sum(dim=-1)  # (B, N)
    entropy_loss = -entropy.mean(dim=1) # (B, )
    
    return entropy_loss

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


class OutlierExposureModule(nn.Module):
    """
    Outlier Exposure module that generates pseudo-outliers by shuffling features
    and encourages uniform predictions for these outliers.
    """
    def __init__(
        self,
        shuffle_ratio: float = 0.3,
        oe_lambda: float = 1.0,
        oe_lambda_memory: float = 0.0,
    ):
        super().__init__()
        self.shuffle_ratio = shuffle_ratio
        self.oe_lambda = oe_lambda
        self.oe_lambda_memory = oe_lambda_memory
        
        print(f"Init OutlierExposureModule: lambda={oe_lambda}, "
              f"shuffle_ratio={shuffle_ratio}, lambda_memory={oe_lambda_memory}")
    
    def shuffle_features(self, x):
        """
        Shuffle a subset of features to break inter-column dependencies
        Args:
            x: (B, F) input features
        Returns:
            x_shuffled: (B, F) with some features shuffled
        """
        batch_size, num_features = x.shape
        x_shuffled = x.clone()
        
        # Select features to shuffle
        num_shuffle = max(1, int(num_features * self.shuffle_ratio))
        shuffle_indices = torch.randperm(num_features, device=x.device)[:num_shuffle]
        
        # Shuffle selected features across batch
        for idx in shuffle_indices:
            perm = torch.randperm(batch_size, device=x.device)
            x_shuffled[:, idx] = x[perm, idx]
        
        return x_shuffled
    
    def compute_oe_loss(self, x, x_hat_shuf, weight_shuf=None):
        """
        Compute outlier exposure loss
        Args:
            x: (B, F) original input features (before shuffling)
            x_hat_shuf: (B, F) reconstructed shuffled features
            weight_shuf: (B, N, M) optional memory weights for shuffled data
        Returns:
            loss_oe: scalar tensor, total OE loss
        """
        # Push shuffled reconstruction toward original (pre-shuffle) samples
        loss_oe = F.mse_loss(x_hat_shuf, x, reduction='none').mean(dim=1)  # (B,)
        
        # Add memory entropy loss if applicable
        if self.oe_lambda_memory != 0 and weight_shuf is not None:
            loss_memory_entropy = compute_entropy_loss(weight_shuf)
            total_loss = self.oe_lambda * loss_oe + self.oe_lambda_memory * loss_memory_entropy
        else:
            total_loss = self.oe_lambda * loss_oe
        
        return total_loss

class OELATTE(nn.Module):
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
        top_k: int = None,
        is_recurrent: bool = False,
        global_decoder_query: bool = False,
        not_use_memory: bool = False,
        not_use_decoder: bool = False,
        use_oe: bool = False,
        oe_lambda: float = 1.0,
        oe_shuffle_ratio: float = 0.3,
        oe_lambda_memory: float = 0.0,
        entropy_loss_weight: float = 0.0,
        latent_loss_weight: float = 0.0,
    ):
        super(OELATTE, self).__init__()
        assert num_latents is not None
        assert num_memories is not None
        if not use_pos_enc_as_query:
            assert not use_mask_token
        if top_k is not None:
            top_k = min(top_k, num_memories)

        print("Init MemPAE with weight_sharing" if is_weight_sharing else "Init MemPAE without weight sharing")

        self.feature_tokenizer = FeatureTokenizer(num_features, hidden_dim)
        self.memory = MemoryUnit(num_memories, hidden_dim, sim_type, temperature, shrink_thred, top_k, num_heads)
        
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
        self.latents_query = nn.Parameter(torch.empty(1, num_latents, hidden_dim))

        if global_decoder_query:
            print(f"Init decoder query of shape {(1, 1, hidden_dim)} and use expanded decoder query as query token.")
            self.decoder_query = nn.Parameter(torch.empty(1, 1, hidden_dim))
        else:
            if use_pos_enc_as_query:
                if use_mask_token:
                    print(f"Init decoder query of shape {(1, 1, hidden_dim)} and use decoder query + pos_encoding as query token.")
                    self.decoder_query = nn.Parameter(torch.empty(1, 1, hidden_dim))
                else:
                    print(f"Do not init decoder query but use positional encoding as decoder query")
                    self.decoder_query = None
            else:
                print(f"Init decoder query of shape {(1, num_features, hidden_dim)}")
                self.decoder_query = nn.Parameter(torch.empty(1, num_features, hidden_dim))

        self.num_features = num_features
        self.is_weight_sharing = is_weight_sharing
        self.use_pos_enc_as_query = use_pos_enc_as_query
        self.use_mask_token = use_mask_token
        self.depth = depth
        self.is_recurrent = is_recurrent
        self.global_decoder_query = global_decoder_query
        self.not_use_memory = not_use_memory
        self.not_use_decoder = not_use_decoder
        self.entropy_loss_weight = entropy_loss_weight
        self.latent_loss_weight = latent_loss_weight
        
        # OE module
        self.use_oe = use_oe
        if use_oe:
            self.oe_module = OutlierExposureModule(
                shuffle_ratio=oe_shuffle_ratio,
                oe_lambda=oe_lambda,
                oe_lambda_memory=oe_lambda_memory
            )
        else:
            self.oe_module = None
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.decoder_query is not None:
            nn.init.trunc_normal_(self.decoder_query, std=0.02)
        nn.init.trunc_normal_(self.latents_query, std=0.02)
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def _process_latents(self, feature_embedding, batch_size):
        """Process feature embeddings through encoder and self-attention blocks"""
        latents_query = self.latents_query.expand(batch_size, -1, -1)
        latents, _ = self.encoder(latents_query, feature_embedding, feature_embedding, return_weight=True)
        
        if self.is_weight_sharing:
            for _ in range(self.depth):
                latents, _ = self.block(latents, return_weight=True)
                if self.is_recurrent:
                    latents, _ = self.memory(latents)
        else:
            for block in self.block:
                latents, _ = block(latents, return_weight=True)
        
        return latents
    
    def _decode_latents(self, latents_hat, decoder_query):
        """Decode latents to reconstruction"""
        if self.not_use_decoder:
            output = latents_hat
        else:
            output, _ = self.decoder(decoder_query, latents_hat, latents_hat, return_weight=True)
        
        x_hat = self.proj(output)
        return x_hat
    
    def _get_decoder_query(self, batch_size):
        """Get decoder query based on configuration"""
        if self.use_pos_enc_as_query:
            if self.use_mask_token:
                decoder_query = self.decoder_query.expand(batch_size, self.num_features, -1)
                decoder_query = decoder_query + self.pos_encoding
            else:
                decoder_query = self.pos_encoding.expand(batch_size, -1, -1)
        else:
            decoder_query = self.decoder_query.expand(batch_size, -1, -1)
        
        return decoder_query

    def forward(self, x):
        batch_size, num_features = x.shape

        # Main forward pass
        feature_embedding = self.feature_tokenizer(x)
        feature_embedding = feature_embedding + self.pos_encoding
        latents = self._process_latents(feature_embedding, batch_size) # (B, N, D)

        # Memory addressing
        if self.not_use_memory:
            latents_hat = latents 
            memory_weight = None
        else:
            latents_hat, memory_weight = self.memory(latents)

        # Decode
        decoder_query = self._get_decoder_query(batch_size)
        x_hat = self._decode_latents(latents_hat, decoder_query)
        
        loss_rec = F.mse_loss(x_hat, x, reduction='none').mean(dim=1)

        if self.entropy_loss_weight > 0: 
            loss_memory_entropy = memory_weight * torch.log(memory_weight + 1e-12)
            loss_memory_entropy = -1.0 * loss_memory_entropy.sum(dim=-1)
            loss_memory_entropy = loss_memory_entropy.mean(dim=-1) * self.entropy_loss_weight

        if self.latent_loss_weight > 0:
            loss_latent_rec = F.mse_loss(latents, latents_hat, reduction='none').mean(dim=[1, 2])
            loss_latent_rec = self.latent_loss_weight * loss_latent_rec
        # print(loss_rec.shape)
        # print(loss_memory_entropy.shape)
        # print(loss_latent_rec.shape)
        anomaly_score = loss_rec + loss_memory_entropy + loss_latent_rec

        # Outlier Exposure (only during training)
        if self.training and self.use_oe:
            # Generate shuffled samples
            x_shuffled = self.oe_module.shuffle_features(x)
            
            # Forward pass on shuffled samples
            feature_embedding_shuf = self.feature_tokenizer(x_shuffled)
            feature_embedding_shuf = feature_embedding_shuf + self.pos_encoding
            latents_shuf = self._process_latents(feature_embedding_shuf, batch_size)
            
            # Memory addressing for shuffled samples
            if not self.not_use_memory:
                latents_hat_shuf, weight_shuf = self.memory(latents_shuf)
            else:
                latents_hat_shuf = latents_shuf
                weight_shuf = None
            
            # Decode shuffled samples
            x_hat_shuf = self._decode_latents(latents_hat_shuf, decoder_query)
            
            # Compute OE loss
            loss_oe = self.oe_module.compute_oe_loss(x, x_hat_shuf, weight_shuf)
            output = anomaly_score + loss_oe
        else:
            output = anomaly_score
    
        return output