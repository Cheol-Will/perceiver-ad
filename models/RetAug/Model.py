import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from itertools import combinations
from sklearn.neighbors import NearestNeighbors


class FeatureEmbedding(nn.Module):
    """Embed each feature with mask token"""
    def __init__(self, num_features: int, hidden_dim: int):
        super().__init__()
        # Each feature gets its own embedding (value + mask_token)
        self.embeddings = nn.ModuleList([
            nn.Linear(2, hidden_dim) for _ in range(num_features)
        ])
        
    def forward(self, x_list):
        """x_list: list of (N, 2) tensors -> (N, D, H)"""
        embedded = [self.embeddings[i](x_list[i]) for i in range(len(x_list))]
        return torch.stack(embedded, dim=1)


class TransformerEncoder(nn.Module):
    """Standard transformer encoder"""
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        x = self.transformer(x)
        return self.norm(x)


class KNNRetrieval(nn.Module):
    """KNN-based retrieval from original paper"""
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        
    def forward(self, query, support):
        """
        query: (N_q, D, H), support: (N_s, D, H)
        Returns: indices (N_q, k), weights (N_q, k)
        """
        # Flatten for distance computation
        q_flat = query.reshape(query.size(0), -1).detach().cpu().numpy()
        s_flat = support.reshape(support.size(0), -1).detach().cpu().numpy()
        
        # KNN search
        nn_model = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        nn_model.fit(s_flat)
        distances, indices = nn_model.kneighbors(q_flat)
        
        # Convert to weights
        similarities = 1.0 / (distances + 1e-6)
        weights = similarities / similarities.sum(axis=1, keepdims=True)
        
        return torch.from_numpy(indices), torch.from_numpy(weights).float()


class AttentionRetrieval(nn.Module):
    """Attention-based retrieval from original paper"""
    def __init__(self, hidden_dim: int, k: int):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, query, support):
        """
        query: (N_q, D, H), support: (N_s, D, H)  
        Returns: indices (N_q, k), weights (N_q, k)
        """
        N_q, D, H = query.shape
        N_s = support.size(0)
        
        # Project and flatten
        q = self.Q(query).reshape(N_q, -1)  # (N_q, D*H)
        k = self.K(support).reshape(N_s, -1)  # (N_s, D*H)
        
        # L2 distance-based similarity (attention_bsim from paper)
        q_exp = q.unsqueeze(1)  # (N_q, 1, D*H)
        k_exp = k.unsqueeze(0)  # (1, N_s, D*H)
        similarity = -torch.norm(q_exp - k_exp, dim=2, p=2) ** 2
        similarity = similarity / math.sqrt(self.hidden_dim)
        
        # Get top-k
        topk_sim, topk_idx = torch.topk(similarity, self.k, dim=1)
        weights = F.softmax(topk_sim, dim=1)
        
        return topk_idx, weights


class RetAug(nn.Module):
    """
    Retrieval-Augmented Anomaly Detection
    Based on original paper implementation
    """
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout_prob: float = 0.1,
        retrieval_type: str = 'knn',  # 'knn' or 'attention'
        retrieval_k: int = 5,
        retrieval_location: str = 'post-encoder',  # 'post-embedding' or 'post-encoder'
        aggregation_lambda: float = 0.5,
        train_mask_prob: float = 0.15,
        test_mask_prob: float = 0.15,
        num_reconstructions: int = 20,
        max_masked_features: int = None,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.train_mask_prob = train_mask_prob
        self.test_mask_prob = test_mask_prob
        self.num_reconstructions = num_reconstructions
        self.retrieval_location = retrieval_location
        self.aggregation_lambda = aggregation_lambda
        
        # Embedding
        self.feature_embedding = FeatureEmbedding(num_features, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, num_features, hidden_dim))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        
        # Encoder
        self.encoder = TransformerEncoder(hidden_dim, num_layers, num_heads, dropout_prob)
        
        # Retrieval
        if retrieval_type == 'knn':
            self.retrieval = KNNRetrieval(k=retrieval_k)
        else:
            self.retrieval = AttentionRetrieval(hidden_dim=hidden_dim, k=retrieval_k)
        
        # Decoder
        self.decoder = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_features)
        ])
        
        # Mask bank
        if max_masked_features is None:
            max_masked_features = max(1, int(num_features * test_mask_prob))
            print(max_masked_features)
        self.max_masked_features = max_masked_features
        self.mask_bank = self._generate_mask_bank(num_features, max_masked_features)
        
        print(f"RetAug Model: {retrieval_type} retrieval at {retrieval_location}")
        print(f"Mask bank: {len(self.mask_bank)} masks")

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
        """Random masking"""
        if mask_prob == 0:
            return torch.zeros_like(x)
        mask = torch.bernoulli(torch.full_like(x, mask_prob))
        # Ensure at least one feature is masked
        no_mask = mask.sum(dim=1) == 0
        if no_mask.any():
            for idx in torch.where(no_mask)[0]:
                mask[idx, torch.randint(0, x.size(1), (1,))] = 1
        return mask
    
    def _prepare_input(self, X, M):
        """Prepare input with mask tokens: (N, D) -> list of (N, 2)"""
        X_masked = X * (1 - M)
        x_list = [
            torch.stack([X_masked[:, i], M[:, i]], dim=1)
            for i in range(X.size(1))
        ]
        return x_list
    
    def aggregate(self, query, support, indices, weights):
        """
        Aggregate query with retrieved samples
        query: (N_q, D, H)
        support: (N_s, D, H)
        indices: (N_q, k)
        weights: (N_q, k)
        """
        N_q = query.size(0)
        device = query.device
        
        # Gather retrieved samples
        indices = indices.to(device)
        weights = weights.to(device)
        
        # (N_q, k, D, H)
        retrieved = support[indices]
        
        # Weighted aggregation
        weights_exp = weights.unsqueeze(-1).unsqueeze(-1)  # (N_q, k, 1, 1)
        retrieved_agg = (retrieved * weights_exp).sum(dim=1)  # (N_q, D, H)
        
        # Linear interpolation
        return (1 - self.aggregation_lambda) * query + self.aggregation_lambda * retrieved_agg
    
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
        
        # Masking
        if mode == 'train':
            M = self._create_mask(X, self.train_mask_prob)
        else:
            if mask_idx is not None:
                M = self.mask_bank[mask_idx].unsqueeze(0).expand(N, -1).to(device)
            else:
                M = self._create_mask(X, self.test_mask_prob)
        
        # Prepare input with mask tokens
        x_list = self._prepare_input(X, M)
        
        # Split support and query
        if query_indices is not None:
            support_mask = torch.ones(N, dtype=torch.bool, device=device)
            support_mask[query_indices] = False
            support_indices = torch.where(support_mask)[0]
        else:
            support_indices = torch.arange(N, device=device)
            query_indices = torch.arange(N, device=device)
        
        # === Forward with Retrieval ===
        
        # 1. Embedding
        H = self.feature_embedding(x_list)  # (N, D, H)
        H = H + self.pos_encoding
        
        # 2. Retrieval and Aggregation based on location
        if self.retrieval_location == 'post-embedding':
            # Retrieve at embedding level
            H_query = H[query_indices]
            H_support = H[support_indices].detach()  # Don't update support
            
            # Retrieve
            indices, weights = self.retrieval(H_query, H_support)
            
            # Aggregate
            H_query_agg = self.aggregate(H_query, H_support, indices, weights)
            
            # Combine
            H_full = H.clone()
            H_full[query_indices] = H_query_agg
            
            # Encode
            H_encoded = self.encoder(H_full)
            
        elif self.retrieval_location == 'post-encoder':
            # Encode first
            H_encoded = self.encoder(H)
            
            # Retrieve at encoded level
            H_query = H_encoded[query_indices]
            H_support = H_encoded[support_indices].detach()
            
            # Retrieve
            indices, weights = self.retrieval(H_query, H_support)
            
            # Aggregate
            H_query_agg = self.aggregate(H_query, H_support, indices, weights)
            
            # Update
            H_encoded = H_encoded.clone()
            H_encoded[query_indices] = H_query_agg
        
        # 3. Decode
        X_hat = torch.stack([
            self.decoder[i](H_encoded[:, i, :]).squeeze(-1)
            for i in range(self.num_features)
        ], dim=1)
        
        # 4. Compute loss on masked positions
        recon_loss = F.mse_loss(X_hat, X, reduction='none')
        masked_loss = recon_loss * M
        sample_loss = masked_loss.sum(dim=1) / M.sum(dim=1).clamp(min=1)
        
        # Extract query loss
        query_loss = sample_loss[query_indices]
        
        if mode == 'train':
            return query_loss.mean()
        else:
            return query_loss