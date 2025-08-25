import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb

class Disent(nn.Module):
    def __init__(self, model_config):
        super(Disent, self).__init__()
        self.model_config = model_config
        dim = model_config['hidden_dim'] 
        att_dim = model_config['data_dim']
        num_heads = 2
        qkv_bias = True

        encoder = []
        encoder_dim = att_dim
        for _ in range(2):
            encoder.append(nn.Linear(encoder_dim, dim * 2, bias=qkv_bias))
            encoder.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = dim * 2
        encoder.append(nn.Linear(encoder_dim, dim, bias=qkv_bias))
        self.encoder = nn.Sequential(*encoder)

        self.att_dim = att_dim
        self.num_heads = num_heads
        self.head_dim = dim
        self.scale = self.head_dim ** -0.5

        decoder = []
        decoder_dim = dim
        for _ in range(2):
            decoder.append(nn.Linear(decoder_dim, dim * 2, bias=qkv_bias))
            decoder.append(nn.LeakyReLU(0.2, inplace=True))
            decoder_dim = dim * 2
        decoder.append(nn.Linear(decoder_dim, att_dim, bias=qkv_bias))
        self.decoder = nn.Sequential(*decoder)

        self.qkv = nn.Linear(dim, dim * 3 * num_heads, bias=qkv_bias)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, inputs):
        # print("shape of inputs:", inputs.shape)
        hidden_feat = self.encoder(inputs)
        # print("shape of hidden_feat:", hidden_feat.shape)

        B, N, C = hidden_feat.shape
        qkv = self.qkv(hidden_feat).reshape(B, N, 3, self.num_heads, C).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print(f"attn shape: {attn.shape}, q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        attn_1, attn_2 = attn.unbind(1)
        attn_1 = attn_1.softmax(dim=-1)
        attn_2 = attn_2.softmax(dim=-1)

        v1, v2 = v.unbind(1)

        z_1 = attn_1 @ v1
        z_2 = attn_2 @ v2

        output_1 = self.decoder(z_1)
        output_2 = self.decoder(z_2)

        if self.training:
            recon_loss = F.mse_loss(inputs, output_1) + F.mse_loss(inputs, output_2)
            dis_loss = torch.mean(self.cos(attn_1.reshape((B, N ** 2)), attn_2.reshape((B, N ** 2))))
            # print("inputs:", inputs.shape, "output_1:", output_1.shape, "output_2:", output_2.shape)
            # print("z_1:", z_1.shape, "z_2:", z_2.shape)
            # print("recon_loss:", recon_loss.shape, "dis_loss:", dis_loss.shape)
            return recon_loss, dis_loss
        else:
            anomaly_score = (F.mse_loss(inputs, output_1, reduction='none') + F.mse_loss(inputs, output_2, reduction='none')).sum(dim=[1,2])
            # print(f"anomaly_score shape: {anomaly_score.shape}")
            return anomaly_score