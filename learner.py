import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
from torch_geometric.utils import get_laplacian


class Learner(nn.Module):
    def __init__(self):
        super(Learner, self).__init__()
        self.weights = nn.ParameterList()

        self.weights.append(Parameter(torch.Tensor(1, 16)))
        self.weights.append(Parameter(torch.zeros(16)))

        self.weights.append(Parameter(torch.Tensor(16, 16)))
        self.weights.append(Parameter(torch.zeros(16)))

        self.weights.append(Parameter(torch.zeros(16, 11)))
        self.weights.append(Parameter(torch.zeros(11)))

        for idx, p in enumerate(self.weights):
            if p.dim() < 2:
                continue
            nn.init.kaiming_normal_(self.weights[idx])

    def graphConv(self, edge_index, x, weight, bias=None):
        edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=x.size(0))
        L = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([x.size(0), x.size(0)]))

        x_hat = torch.sparse.mm(L, x)
        x_hat = x_hat @ weight
        if bias is not None:
            x_hat = x_hat + bias
        return x_hat

    def forward(self, data, vars=None):
        if vars is None:
            vars = self.weights

        ret = []
        for i in range(data.y.size(0)):
            graph = data[i]
            x, edge_idx = graph.x, graph.edge_index
            x = F.relu(self.graphConv(edge_idx, x, vars[0], vars[1]), inplace=False)
            x = F.relu(self.graphConv(edge_idx, x, vars[2], vars[3]), inplace=False)
            pooled_feature = torch.mean(x, dim=0)
            pooled_feature = torch.matmul(pooled_feature, vars[4]).squeeze() + vars[5]
            ret.append(pooled_feature.view(1, -1))

        re = torch.cat(ret, dim=0)
        return re

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.weights:
                    if p is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self, recurse=True):
        return self.weights
