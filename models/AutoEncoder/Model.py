import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(
        self, 
        num_features: int,
        hidden_dim: int, 
        depth: int,
    ):
        super().__init__()
        print('Initialize custom AutoEncoder.')

        encoder = []
        encoder.append(nn.Linear(num_features, hidden_dim))
        encoder.append(nn.ReLU())
        for _ in range(depth - 1):
            encoder.append(nn.Linear(hidden_dim, hidden_dim))
            encoder.append(nn.ReLU())
        
        decoder = []
        for _ in range(depth - 1):
            decoder.append(nn.Linear(hidden_dim, hidden_dim))
            decoder.append(nn.ReLU())
        decoder.append(nn.Linear(hidden_dim, num_features))
        
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)


    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x, reduction='none').mean(dim=1)
        return loss