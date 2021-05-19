import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MEANAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_num):
        super(MEANAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_num = hidden_num
        self.hidden_num_mid = 96
        self.encoder = nn.Sequential(
            nn.Linear(input_size, self.hidden_num_mid),
            nn.ReLU(),
            nn.Linear(self.hidden_num_mid, self.hidden_num),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_num, self.hidden_num_mid),
            nn.ReLU(),
            nn.Linear(self.hidden_num_mid, self.input_size),
            nn.ReLU()
        )
        self.last_fc = nn.Linear(self.hidden_num, self.hidden_num)

    def forward(self, inputs):
        emb = self.encoder(inputs)
        emb_pool = torch.mean(emb, dim=0, keepdim=True)
        emb_all = self.last_fc(emb_pool)
        emb_dec = self.decoder(emb)
        loss = F.mse_loss(emb_dec, inputs)
        return emb_all, loss


class LSTMAutoencoder(nn.Module):
    def __init__(self, hidden_num, reverse=True):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_num = hidden_num
        self.reverse = reverse
        self.encoder = nn.GRU(
            hidden_size=self.hidden_num,
            batch_first=True
        )
        self.decoder = nn.GRU(
            hidden_size=self.hidden_num,
            batch_first=True
        )

    def forward(self, inputs, h_state=None):
        inputs = torch.unsqueeze(inputs, 0)
        z_codes, enc_state = self.encoder(inputs, h_state)

        dec_state = self.enc_state
        dec_input = torch.zeros(inputs[0].size())

        dec_out = []



