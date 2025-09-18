import torch
import torch.nn as nn
import numpy as np 


class Generator(nn.Module):
    def __init__(
        self, 
        model: nn.Module, 
        num_features: int, 
        mask_generator_depth: int,
        mask_num: int,
    ):
        super(Generator, self).__init__()
        self.masks = model._make_nets(num_features, mask_generator_depth, mask_num)
        self.mask_num = mask_num

    def forward(self, x):
        x = x
        x_T = torch.empty(x.shape[0], self.mask_num, x.shape[-1]).to(x)
        masks = []
        for i in range(self.mask_num):
            mask = self.masks[i](x)
            masks.append(mask.unsqueeze(1))
            mask = torch.sigmoid(mask)
            x_T[:, i] = mask * x
        masks = torch.cat(masks, axis=1)
        return x_T, masks


class SingleNet(nn.Module):
    def __init__(self, num_features, hidden_dim, depth):
        super(SingleNet, self).__init__()
        net = []
        input_dim = num_features
        for _ in range(depth-1):
            net.append(nn.Linear(input_dim, hidden_dim, bias=False))
            net.append(nn.ReLU())
            input_dim= hidden_dim
        net.append(nn.Linear(input_dim, num_features, bias=False))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)
        return out


class MultiNets():
    def _make_nets(self, num_features, detph, mask_num):
        multinets = nn.ModuleList(
            [SingleNet(num_features, num_features, detph) for _ in range(mask_num)]) # MCM uses num_features as hidden_dim.
        return multinets

