import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import Sequential, GCNConv, global_mean_pool


class GraphEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(GraphEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = Sequential('x, edge_index', [
            (GCNConv(in_channels, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(64, 64), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channels)
        ])

    def forward(self, x, edge_index, batch):
        x_encode = self.encoder(x, edge_index)
        xpool = global_mean_pool(x_encode, batch)
        return xpool
