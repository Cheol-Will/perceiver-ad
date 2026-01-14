import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import math
import random
from models.layers import BaseDecoder, BaseEncoder


class MomentumQueue(nn.Module):
    """
    Memory Queue with exponential moving average update.
    """
    def __init__(
        self, 
        dim, 
        queue_size=1024, 
        temperature=0.1,
    ):
        super(MomentumQueue, self).__init__()
        # Queue: (queue_size, dim)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue", torch.randn(queue_size, dim))
        self.queue = F.normalize(self.queue, dim=-1)
        
        self.dim = dim
        self.queue_size = queue_size
        self.temperature = temperature

    def reset_parameters(self):
        self.queue = F.normalize(torch.randn(self.queue_size, self.dim, device=self.queue.device), dim=-1)
        self.queue_ptr[0] = 0

    @torch.no_grad()
    def enqueue_dequeue(self, keys):
        """
        FIFO queue update.
        """
            
        batch_size = keys.shape[0]
        assert batch_size <= self.queue_size

        ptr = int(self.queue_ptr)
        
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


    def forward(self, query, key):
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        
        l_pos = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1)
        l_neg = torch.einsum("nc,kc->nk", [query, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self.enqueue_dequeue(key)

        return logits, labels


class MOCO(nn.Module):
    """
    MOCO based Anomaly Detection Model.
    """
    def __init__(self, 
        num_features,
        hidden_dim,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout_prob=0.0,
        queue_size=1024,
        momentum=0.999,
        temperature=1.0,
        use_flash_attn: bool = False,
        mixup_alpha: float = 1.0, 
        contrastive_loss_weight: float = 1.0, 
    ):
        super(MOCO, self).__init__()

        self.encoder_q = BaseEncoder(num_features, hidden_dim, depth, num_heads, mlp_ratio, dropout_prob, use_flash_attn)
        self.encoder_k = BaseEncoder(num_features, hidden_dim, depth, num_heads, mlp_ratio, dropout_prob, use_flash_attn)
        self.momentum_queue = MomentumQueue(hidden_dim, queue_size, temperature)
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
        self.mixup_alpha = mixup_alpha
        self.contrastive_loss_weight = contrastive_loss_weight


    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    def forward(self, x, return_dict=False):
        batch_size = x.shape[0]
        query = self.encoder_q(x)

        if self.training:
            with torch.no_grad():
                self._momentum_update_key_encoder()
                mixup_index = torch.randint(0, batch_size, (batch_size,))
                x_key = self.mixup_alpha * x + (1 - self.mixup_alpha) * x[mixup_index]
                key = self.encoder_k(x_key)

            logits, labels = self.momentum_queue(query, key)
            contrastive_loss = F.cross_entropy(logits, labels)
        else:
            contrastive_loss = 0

        x_hat = self.decoder(query, self.pos_encoding)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=-1)
        
        loss = reconstruction_loss.mean() + contrastive_loss * self.contrastive_loss_weight
        anomaly_score = reconstruction_loss
        
        if return_dict:
            return {
                'loss': loss,
                'reconstruction_loss': reconstruction_loss.mean(),
                'contrastive_loss': contrastive_loss,
                'anomaly_score': anomaly_score,
                'x_hat': x_hat,
                'query': query,
            }
        return anomaly_score