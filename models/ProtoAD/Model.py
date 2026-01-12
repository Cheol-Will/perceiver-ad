import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import BaseDecoder, BaseEncoder


class PrototypeContrastive(nn.Module):
    def __init__(
        self,
        dim: int,
        num_prototypes: int = 10,
        temperature: float = 0.1,
        sinkhorn_iters: int = 3,
        sinkhorn_eps: float = 0.05,
    ):
        super().__init__()
        
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_eps = sinkhorn_eps
        self.dim = dim
        
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.trunc_normal_(self.prototypes, std=0.02)
    
    @torch.no_grad()
    def sinkhorn_knopp(self, scores: torch.Tensor) -> torch.Tensor:
        Q = torch.exp(scores / self.sinkhorn_eps)
        Q /= Q.sum()
        
        K, B = self.num_prototypes, scores.shape[0]
        
        for _ in range(self.sinkhorn_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= K
            Q /= Q.sum(dim=1, keepdim=True)
        
        return Q * B
    
    def forward(self, z: torch.Tensor):
        """
        Returns:
            contrastive_loss: scalar (Main Loss)
            entropy: (batch_size,) for anomaly scoring
        """
        z_norm = F.normalize(z, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        
        similarity = z_norm @ proto_norm.T  # (batch_size, num_prototypes)
        
        # Sinkhorn target
        with torch.no_grad():
            Q = self.sinkhorn_knopp(similarity.detach())
        
        # Contrastive loss (Main)
        log_probs = F.log_softmax(similarity / self.temperature, dim=-1)
        contrastive_loss = -(Q * log_probs).sum(dim=-1).mean()
        
        # Entropy for anomaly scoring
        probs = F.softmax(similarity / self.temperature, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        
        return contrastive_loss, entropy


class PrototypeAD(nn.Module):
    """
    Prototype-based Anomaly Detection
    """
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout_prob: float = 0.0,
        num_prototypes: int = 10,
        temperature: float = 0.1,
        sinkhorn_iters: int = 3,
        sinkhorn_eps: float = 0.05,
        contrastive_loss_weight: float = 0.1,  #
        use_flash_attn: bool = False,
        anomaly_score_type: str = 'reconstruction',  # 'entropy' or 'reconstruction'
    ):
        super().__init__()
        
        self.encoder = BaseEncoder(
            num_features, hidden_dim, depth, num_heads, 
            mlp_ratio, dropout_prob, use_flash_attn
        )
        self.decoder = BaseDecoder(
            num_features, hidden_dim, num_heads, mlp_ratio, dropout_prob
        )
        self.pos_encoding = nn.Parameter(torch.empty(1, num_features, hidden_dim))
        
        self.prototype_module = PrototypeContrastive(
            dim=hidden_dim,
            num_prototypes=num_prototypes,
            temperature=temperature,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_eps=sinkhorn_eps,
        )
        
        self.contrastive_loss_weight = contrastive_loss_weight
        self.anomaly_score_type = anomaly_score_type
        self.hidden_dim = hidden_dim
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
    
    def forward(self, x: torch.Tensor, return_dict: bool = False):
        # Encode
        z = self.encoder(x)
        
        # Prototype contrastive (Main)
        if self.training:
            contrastive_loss, entropy = self.prototype_module(z)
        else:
            with torch.no_grad():
                contrastive_loss, entropy = self.prototype_module(z)
            contrastive_loss = torch.tensor(0.0, device=x.device)
        
        # Decode (Auxiliary)
        x_hat = self.decoder(z, self.pos_encoding)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=-1)
        loss = self.contrastive_loss_weight * contrastive_loss + reconstruction_loss.mean()
        
        # Anomaly score
        if self.anomaly_score_type == 'entropy':
            anomaly_score = entropy
        else:  # 'reconstruction'
            anomaly_score = reconstruction_loss
        
        if return_dict:
            return {
                'loss': loss,
                'contrastive_loss': contrastive_loss,  # Main
                'reconstruction_loss': reconstruction_loss.mean(),  # Auxiliary
                'anomaly_score': anomaly_score,
                'entropy': entropy,
                'x_hat': x_hat,
                'z': z,
            }
        return anomaly_score