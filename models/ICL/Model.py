import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ICL(nn.Module):
    def __init__(self, model_config):
        super(ICL, self).__init__()
        self.data_dim = model_config['data_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.rep_dim = model_config['rep_dim']
        self.tau = model_config['temperature']
        self.max_negatives = model_config['max_negatives']

        if self.data_dim <= 40:
            self.kernel_size = 2
        elif 40 < self.data_dim <= 160:
            self.kernel_size = 10
        elif 160 < self.data_dim <= 240:
            self.kernel_size = self.data_dim - 150
        elif 240 < self.data_dim <= 480:
            self.kernel_size = self.data_dim - 200
        else:
            self.kernel_size = self.data_dim - 400

        self.en_nlayers = model_config['en_nlayers']
        self.de_nlayers = model_config['de_nlayers']
        self.model_config = model_config
    
        phi = []
        encoder_dim = self.data_dim
        for _ in range(self.en_nlayers-2):
            phi.append(nn.Linear(encoder_dim,self.hidden_dim,bias=False))
            phi.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = self.hidden_dim
        phi.append(nn.Linear(encoder_dim,model_config['basis_vector_num'],bias=False))
        self.phi = nn.Sequential(*phi)

        encoder = []
        encoder_dim = self.data_dim
        for _ in range(self.en_nlayers-1):
            encoder.append(nn.Linear(encoder_dim,self.hidden_dim,bias=False))
            encoder.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = self.hidden_dim
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        # for _ in range(self.de_nlayers-1):
        #     decoder.append(nn.Linear(self.hidden_dim,self.hidden_dim,bias=False))
        #     decoder.append(nn.LeakyReLU(0.2, inplace=True))
        decoder.append(nn.Linear(self.hidden_dim,self.data_dim,bias=False))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x_input):
        h = self.encoder(x_input)

        weight = F.softmax(self.phi(x_input), dim=1)
        h_ = weight@self.basis_vector

        mse = F.mse_loss(h, h_, reduction='none')

        return mse.sum(dim=1,keepdim=True)
    