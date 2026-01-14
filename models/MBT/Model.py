import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import math
import random
from models.layers import BaseDecoder, BaseEncoder


class MemoryBank(nn.Module):
    """
    Memory Bank for tabular anomaly detection.
    """
    def __init__(self, dim, temperature=0.1, top_k=5):
        super(MemoryBank, self).__init__()
        self.dim = dim
        self.temperature = temperature
        self.top_k = top_k # set 0 for averaging all entries.
        
        self.register_buffer("memory", None)
        self.register_buffer("index_map", None)
        self.register_buffer("num_samples", torch.zeros(1, dtype=torch.long))
        
        self._is_built = False
        
    def reset(self):
        """Reset memory bank"""
        self.memory = None
        self.index_map = None
        self.num_samples[0] = 0
        self._is_built = False

    def is_built(self):
        """Check if memory bank is built"""
        return self._is_built and self.memory is not None

    def get_size(self):
        """Get current memory bank size"""
        return int(self.num_samples) if self.is_built() else 0

    @torch.no_grad()
    def build(self, keys, sample_indices):
        """
        Build memory bank from all training samples.
        Called at the start of each epoch.
        """
        num_samples = keys.shape[0]
        
        # Normalize and store
        self.memory = F.normalize(keys.clone(), dim=-1)
        self.index_map = sample_indices.clone()
        self.num_samples[0] = num_samples
        self._is_built = True

    @torch.no_grad()
    def retrieve(self, query, exclude_indices=None):
        """
        Retrieve closest vectors from memory bank with top-k and temperature.
        """
        if not self.is_built():
            raise RuntimeError("Memory bank not built. Call build() first.")
            
        batch_size = query.shape[0]
        device = query.device
        num_samples = int(self.num_samples)
        
        # Normalize query and compute cosine similarity: (B, N)
        query_norm = F.normalize(query, dim=-1)
        similarity = torch.matmul(query_norm, self.memory.T)
        
        # Apply exclusion mask if provided (for training self-exclusion)
        if exclude_indices is not None:
            # Set similarity to -inf for excluded indices
            exclude_mask = (exclude_indices.unsqueeze(1) == self.index_map.unsqueeze(0))  # (B, N)
            similarity = similarity.masked_fill(exclude_mask, float('-inf'))
        
        # Determine effective top_k
        # top_k=0 means use all memory
        if self.top_k == 0:
            effective_top_k = num_samples
            if exclude_indices is not None:
                effective_top_k = num_samples - 1
        else:
            effective_top_k = min(self.top_k, num_samples)
            if exclude_indices is not None:
                effective_top_k = min(effective_top_k, num_samples - 1)
        effective_top_k = max(1, effective_top_k)
        
        # Get top-k similarities and indices
        top_k_sim, top_k_indices = torch.topk(similarity, effective_top_k, dim=-1)  # (B, K)
        top_k_weights = F.softmax(top_k_sim / self.temperature, dim=-1)  # (B, K)
        top_k_vectors = self.memory[top_k_indices]
        
        # Weighted sum of top-k vectors 
        retrieved = (top_k_weights.unsqueeze(-1) * top_k_vectors).sum(dim=1)
        retrieved = F.normalize(retrieved, dim=-1)
        
        # Distance to closest vector (for future use)
        max_similarity = top_k_sim[:, 0]  # (B,)
        distances = 1 - max_similarity
        
        return retrieved, distances, top_k_weights

    def forward(self, query, exclude_indices=None):
        return self.retrieve(query, exclude_indices)


class MBT(nn.Module):
    """
    Memory Bank Transformer for Tabular Anomaly Detection.
    """
    def __init__(self, 
        num_features,
        hidden_dim,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout_prob: float = 0.0,
        use_distance_score: bool = False,
        temperature: float = 0.1,
        top_k: int = 5,  # 0 means use all memory
        distance_weight: float = 0.0,
        use_flash_attn: bool = False,
    ):
        super(MBT, self).__init__()
        self.use_distance_score = use_distance_score
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.top_k = top_k
        self.distance_weight = distance_weight
        
        self.encoder = BaseEncoder(num_features, hidden_dim, depth, num_heads, mlp_ratio, dropout_prob, use_flash_attn)
        self.memory_bank = MemoryBank(hidden_dim, temperature, top_k)
        self.decoder = BaseDecoder(num_features, hidden_dim, num_heads, mlp_ratio, dropout_prob)
        
        self.pos_encoding = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        self.reset_parameters()
        
        top_k_str = "all" if top_k == 0 else str(top_k)
        print(f"MBT Model initialized:")
        print(f"  - hidden_dim: {hidden_dim}")
        print(f"  - temperature: {temperature}")
        print(f"  - top_k: {top_k_str}")
        print(f"  - distance_weight: {distance_weight}")

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def set_temperature(self, temperature):
        """Update temperature for retrieval"""
        self.temperature = temperature
        self.memory_bank.temperature = temperature
        
    def set_top_k(self, top_k):
        """Update top-k for retrieval (0 means all)"""
        self.top_k = top_k
        self.memory_bank.top_k = top_k

    def get_memory_bank_size(self):
        """Get current memory bank size"""
        return self.memory_bank.get_size()

    @torch.no_grad()
    def build_memory_bank(self, train_loader, device, use_amp=False):
        was_training = self.training
        self.eval()
        
        all_keys = []
        all_indices = []
        
        for batch in train_loader:
            if len(batch) == 3:
                x_input, _, indices = batch
            elif len(batch) == 2:
                x_input, indices = batch
            else:
                raise ValueError("DataLoader must provide (x, label, index) or (x, index)")
            
            x_input = x_input.to(device)
            
            if use_amp:
                with autocast():
                    key = self.encoder(x_input)
                key = key.float()
            else:
                key = self.encoder(x_input)
            
            # Move to cpu
            all_keys.append(key.cpu())
            all_indices.append(indices)
        
        # concat and cpu
        all_keys = torch.cat(all_keys, dim=0).to(device)
        all_indices = torch.cat(all_indices, dim=0).to(device)
        
        self.memory_bank.build(all_keys, all_indices)
        
        del all_keys, all_indices
        torch.cuda.empty_cache()

        if was_training:
            self.train()

    def forward(self, x, sample_indices=None, return_dict=False):
        """
        Forward pass.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Encode query
        query = self.encoder(x)  # (B, D)
        
        # Check if memory bank is built
        if not self.memory_bank.is_built():
            # Memory bank not built yet - use query directly for decoding
            x_hat = self.decoder(query, self.pos_encoding)
            reconstruction_loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=-1)
            
            if return_dict:
                return {
                    'loss': reconstruction_loss.mean(),
                    'reconstruction_loss': reconstruction_loss.mean(),
                    'distance_loss': torch.tensor(0.0, device=device),
                    'anomaly_score': reconstruction_loss,
                    'distance_score': torch.zeros(batch_size, device=device),
                    'x_hat': x_hat,
                    'query': query,
                    'retrieved': None,
                    'weights': None,
                    'memory_bank_size': 0,
                }
            else:
                return reconstruction_loss
        
        # Retrieve from memory bank
        # During training: exclude self from retrieval using sample_indices
        # During eval: no exclusion needed
        exclude = sample_indices if self.training else None
        retrieved, distances, weights = self.memory_bank.retrieve(query, exclude_indices=exclude)
        
        # Decode using retrieved representation
        x_hat = self.decoder(retrieved, self.pos_encoding)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=-1)  # (B,)
        
        # Total loss: reconstruction + optional distance regularization
        loss = reconstruction_loss.mean() + self.distance_weight * distances.mean()
        
        # Anomaly score
        if self.use_distance_score:
            anomaly_score = reconstruction_loss + distances
        else:
            anomaly_score = reconstruction_loss
        
        if return_dict:
            return {
                'loss': loss,
                'reconstruction_loss': reconstruction_loss.mean(),
                'distance_loss': distances.mean(),
                'anomaly_score': anomaly_score,
                'distance_score': distances,
                'x_hat': x_hat,
                'query': query,
                'retrieved': retrieved,
                'weights': weights,
                'memory_bank_size': self.memory_bank.get_size(),
            }
        else:
            return anomaly_score

    @torch.no_grad()
    def compute_anomaly_scores(self, x, return_components=False):
        """
        Compute anomaly scores for evaluation.
        """
        was_training = self.training
        self.eval()
        
        result = self.forward(x, sample_indices=None, return_dict=True)
        
        if was_training:
            self.train()
        
        if return_components:
            return {
                'score': result['anomaly_score'],
                'e_rec': result['reconstruction_loss'],
                'e_dist': result['distance_score'],
                'x_hat': result['x_hat'],
            }
        else:
            return result['anomaly_score']