from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerStack(nn.Module):

    def __init__(self, layer, N):
        super(LayerStack, self).__init__()
        self.layers = clone(layer, N)

    def forward(self, x, masks):
        for layer in self.layers:
            x = layer(x, masks)
        return x


def clone(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class PositionalEncoder(nn.Module):

    def __init__(self, d_model, dout_p, seq_len = 3660):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dout_p)

        pos_enc_mat = np.zeros((seq_len, d_model))
        odds = np.arange(0, d_model, 2)
        evens = np.arange(1, d_model, 2)

        for pos in range(seq_len):
            pos_enc_mat[pos, odds] = np.sin(pos / (10000 ** (odds / d_model)))
            pos_enc_mat[pos, evens] = np.cos(pos / (10000 ** (evens / d_model)))

        self.pos_enc_mat = torch.from_numpy(pos_enc_mat).unsqueeze(0)

    def forward(self, x):
        B, S, d_model = x.shape
        # torch.cuda.FloatTensor torch.FloatTensor
        x = x + self.pos_enc_mat[:, :S, :].type_as(x)
        x = self.dropout(x)
        # same as input
        return x


class ResidualConnection(nn.Module):

    def __init__(self, size, dout_p):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dout_p)

    def forward(self, x, sublayer):
        # x (B, S, D)
        res = self.norm(x)
        res = sublayer(res)
        res = self.dropout(res)

        return x + res


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dout_p):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dout_p = dout_p
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dout_p)

    def forward(self, x):
        '''In, Out: (B, S, D)'''
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
