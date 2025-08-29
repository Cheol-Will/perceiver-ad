import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MemoryUnit(nn.Module):
    """
      z: (B, D), memories: (N, D)
      sim_type: "cos" | "l2"
    """
    def __init__(
        self,
        num_memories: int,
        hidden_dim: int,
        sim_type: str,    # "cos" or "l2"
        temperature: float = 1.0,
    ):
        super().__init__()
        assert sim_type.lower() in ['cos', 'l2']
        print(f"Init MemoryUnit with {sim_type}")
        self.memories = nn.Parameter(torch.empty(num_memories, hidden_dim))
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.sim_type = sim_type.lower()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.memories)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D), memories: (N, D)
        if self.sim_type == "cos":
            x_norm = F.normalize(x, dim=-1)                  # (B, D)
            mem_norm = F.normalize(self.memories, dim=-1)    # (N, D)
            logits = x_norm @ mem_norm.t()                   # (B, N)

        elif self.sim_type == "l2":
            x_sq = (x ** 2).sum(dim=1, keepdim=True)             # (B, 1)
            m_sq = (self.memories ** 2).sum(dim=1, keepdim=True).t()  # (1, N)
            dist_sq = x_sq + m_sq - 2 * (x @ self.memories.t())  # (B, N)
            dist_sq = dist_sq.clamp_min(0.) 
            logits = -dist_sq                                    

        else:
            raise ValueError(f"sim_type must be 'cos' or 'l2', got {self.sim_type}")

        logits = logits / self.temperature
        weight = F.softmax(logits, dim=-1)                       # (B, N)
        read = weight @ self.memories                            # (B, D)
        return read

def make_mlp(sizes, bias=False, act=nn.LeakyReLU(0.2, inplace=True), last_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1], bias=bias))
        if i < len(sizes) - 2:
            layers.append(act if act is not None else nn.Identity())
        elif last_act is not None:
            layers.append(last_act)
    return nn.Sequential(*layers)


class MemAE(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.data_dim = model_config['data_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.en_nlayers = model_config['depth']
        self.de_nlayers = model_config['depth']
        self.num_memories = model_config['num_memories']
        self.temperature = model_config['temperature'] 
        self.sim_type = model_config['sim_type']

        if self.en_nlayers == 1:
            en_sizes = [self.data_dim, self.hidden_dim]
        else:
            en_sizes = [self.data_dim] + [self.hidden_dim] * self.en_nlayers
        self.encoder = make_mlp(en_sizes, bias=False)

        self.memory = MemoryUnit(self.num_memories, self.hidden_dim, temperature=self.temperature, sim_type=self.sim_type)

        if self.de_nlayers == 1:
            de_sizes = [self.hidden_dim, self.data_dim]
        else:
            de_sizes = [self.hidden_dim] * self.de_nlayers + [self.data_dim]
        self.decoder = make_mlp(de_sizes, bias=False, last_act=None)

    def forward(self, x: torch.Tensor, return_pred = False):
        """
        x: (B, F)
        return:
          x_hat: (B, F)
          loss_per_sample: (B,)
        """
        z = self.encoder(x)                 # (B, D)
        z_hat = self.memory(z)                  # (B, D)

        x_hat = self.decoder(z_hat)       # (B, F)

        mse_per_dim = F.mse_loss(x_hat, x, reduction='none')   # (B, F)
        batch_losses = mse_per_dim.mean(dim=1)              # (B,)

        if return_pred: 
            return batch_losses, x, x_hat
        else:
            return batch_losses