import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from models.layers import FeatureTokenizer, OutputProjection, SelfAttention 


class MomentumQueue(nn.Module):
    """
    Memory Queue with exponential moving average update.
    """
    def __init__(
        self, 
        dim, 
        queue_size=1024, 
        top_k=5,    
        temperature=0.1,
    ):
        super(MomentumQueue, self).__init__()
        # Queue: (queue_size, dim)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue", torch.randn(queue_size, dim))
        self.queue = F.normalize(self.queue, dim=-1)
        
        self.dim = dim
        self.queue_size = queue_size
        self.is_frozen = False # for eval
        self.top_k = top_k
        self.temperature = temperature

    def reset_parameters(self):
        self.queue = F.normalize(torch.randn(self.queue_size, self.dim, device=self.queue.device), dim=-1)
        self.queue_ptr[0] = 0
        self.is_frozen = False

    def freeze_queue(self):
        """Freeze queue after training - no more updates"""
        self.is_frozen = True

    def unfreeze_queue(self):
        """Unfreeze queue to allow updates again"""
        self.is_frozen = False

    @torch.no_grad()
    def enqueue_dequeue(self, keys):
        """
        FIFO queue update.
        
        Args:
            keys: (B, D) - batch of key vectors to add to queue
        """
        if self.is_frozen:
            return
            
        batch_size = keys.shape[0]
        assert batch_size <= self.queue_size

        ptr = int(self.queue_ptr)
        keys = F.normalize(keys, dim=-1)
        
        # Add to queue with wraparound
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[ptr:] = keys[:remaining]
            self.queue[:batch_size - remaining] = keys[remaining:]
        
        # Update pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def retrieve(self, query):
        """
        Retrieve closest vectors from queue.
        
        Args:
            query: (B, D) - query vectors
        Returns:
            retrieved: (B, D) - weighted sum of top-k closest vectors
            distances: (B,) - distances to closest vectors
            top_k_weights: (B, K) - weights for top-k vectors
        """
        batch_size = query.shape[0]
        device = query.device
        
        # Normalize query and compute cosine similarity: (B, queue_size)
        query_norm = F.normalize(query, dim=-1)
        similarity = torch.matmul(query_norm, self.queue.T)
        
        if self.top_k == 0:
            effective_top_k = self.queue_size
        else:
            effective_top_k = min(self.top_k, self.queue_size)
        effective_top_k = max(1, effective_top_k)

        # Get top-k 
        top_k_sim, top_k_indices = torch.topk(similarity, effective_top_k, dim=-1) # (B, K)
        top_k_weights = F.softmax(top_k_sim / self.temperature, dim=-1) # (B, K)
        top_k_vectors = self.queue[top_k_indices]
        
        # Weighted sum of top-k vectors
        retrieved = (top_k_weights.unsqueeze(-1) * top_k_vectors).sum(dim=1)
        retrieved = F.normalize(retrieved, dim=-1)

        # Distance to closest vector
        max_similarity = top_k_sim[:, 0]  # (B,)
        distances = 1 - max_similarity
        
        return retrieved, distances, top_k_weights

    def forward(self, query):
        retrieved, distances, weights = self.retrieve(query)
        return retrieved, distances, weights


class BaseEncoder(nn.Module):
    def __init__(self, 
        num_features,
        hidden_dim,
        depth,
        num_heads=8,
        mlp_ratio=4.0,
        dropout_prob=0.0,
    ):
        super(BaseEncoder, self).__init__()
        self.feature_tokenizer = FeatureTokenizer(num_features, hidden_dim)
        self.encoder = nn.Sequential(*[
            SelfAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob) 
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
    ):
        super(BaseDecoder, self).__init__()
        self.decoder = SelfAttention(hidden_dim, num_heads, mlp_ratio, dropout_prob)
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


class MQ(nn.Module):
    """
    Memory Queue based Anomaly Detection Model.
    
    Key features:
    - Separate evaluation memory bank from training queue
    - Full memory bank construction for proper evaluation
    """
    def __init__(self, 
        num_features,
        hidden_dim,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout_prob=0.0,
        use_distance_score=False,
        queue_size=1024,
        momentum=0.999,
        top_k=5, 
        temperature=1.0,
    ):
        super(MQ, self).__init__()

        self.encoder_q = BaseEncoder(num_features, hidden_dim, depth, num_heads, mlp_ratio, dropout_prob)
        self.encoder_k = BaseEncoder(num_features, hidden_dim, depth, num_heads, mlp_ratio, dropout_prob)
        self.memory_queue = MomentumQueue(hidden_dim, queue_size, top_k, temperature)
        self.decoder = BaseDecoder(num_features, hidden_dim, num_heads, mlp_ratio, dropout_prob)
        self.pos_encoding = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        self.reset_parameters()

        # Initialize encoder_k with encoder_q weights
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.momentum = momentum
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.top_k = top_k
        self.use_distance_score = use_distance_score
        self._eval_memory_bank = None


    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
  
    @torch.no_grad()
    def build_eval_memory_bank(self, train_loader, device):
        """
        Build full memory bank from ALL training samples for evaluation.
        """
        self.eval()
        all_keys = []
        
        for batch in train_loader:
            if len(batch) == 3:
                x_input, _, _ = batch
            else:
                x_input, _ = batch
            
            x_input = x_input.to(device)
            key = self.encoder_k(x_input)
            all_keys.append(key)
        
        eval_memory = F.normalize(torch.cat(all_keys, dim=0), dim=-1)
        eval_size = eval_memory.shape[0]
        
        self._eval_memory_bank = MomentumQueue(
            dim=self.hidden_dim,
            queue_size=eval_size,
            top_k=self.top_k,
            temperature=self.temperature
        ).to(device)
        
        self._eval_memory_bank.enqueue_dequeue(eval_memory)
        self._eval_memory_bank.freeze_queue()
        
        print(f"[Eval] Built full memory bank with {eval_size} samples")

    @torch.no_grad()
    def clear_eval_memory_bank(self):
        """Clear evaluation memory bank"""
        self._eval_memory_bank = None

    def forward(self, x, return_dict=False, use_eval_memory=False):
        batch_size = x.shape[0]
        query = self.encoder_q(x)
        
        queue = self._eval_memory_bank if (use_eval_memory and self._eval_memory_bank is not None) else self.memory_queue
        retrieved, distances, weights = queue.retrieve(query)
        
        x_hat = self.decoder(retrieved, self.pos_encoding)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=-1)
        
        if self.training:
            with torch.no_grad():
                self._momentum_update_key_encoder()
                key = self.encoder_k(x)
                self.memory_queue.enqueue_dequeue(key)
        
        loss = reconstruction_loss.mean()
        anomaly_score = reconstruction_loss + distances if self.use_distance_score else reconstruction_loss
        
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
            }
        return anomaly_score