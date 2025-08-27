import torch
import torch.nn as nn
import torch.nn.functional as F

def make_mlp(sizes, bias=False, act=None, last_act=None):
    if act is None:
        def act_fn(): return nn.LeakyReLU(0.2, inplace=True)
    else:
        def act_fn(): return act if isinstance(act, nn.Module) else act()

    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1], bias=bias))
        if i < len(sizes) - 2:
            layers.append(act_fn())
        elif last_act is not None:
            layers.append(last_act)
    return nn.Sequential(*layers)


class RINMLP(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.data_dim   = model_config['data_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.en_nlayers = model_config['depth']
        self.de_nlayers = model_config['depth']
        self.num_repeat = model_config['num_repeat']

        # Encoder dims: [data_dim, hidden_dim, hidden_dim/2, ...]
        en_dims = [self.data_dim]
        for i in range(self.en_nlayers):
            denom = 2 ** i
            width = max(1, self.hidden_dim // denom)  
            en_dims.append(width)
        self.encoder = make_mlp(en_dims, bias=False)

        # Decoder dims: [..., hidden_dim/2, hidden_dim, data_dim]
        de_dims = list(reversed(en_dims))  
        self.decoder = make_mlp(de_dims, bias=False, last_act=None)

    def forward(self, x: torch.Tensor, return_repeat_losses: bool = False):
        """
        return aggregated reconstruction error for each step.
        """
        # batch_size, num_features = x.shape
        x_hat = x
        losses = []

        for _ in range(self.num_repeat):
            z = self.encoder(x_hat)     # (B, h)
            x_hat = self.decoder(z)     # (B, F)

            mse_per_dim = F.mse_loss(x_hat, x, reduction='none')  # (B, F)
            loss = mse_per_dim.mean(dim=1)                        # (B,)
            losses.append(loss)

        losses = torch.stack(losses, dim=1)   # (B, R)
        batch_losses = losses.mean(dim=1)     # (B,)

        if return_repeat_losses:
            return batch_losses, losses
        else:
            return batch_losses