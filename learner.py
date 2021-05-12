import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch.nn.parameter import Parameter
from copy import deepcopy


class Learner(nn.Module):
    def __init__(self):
        super(Learner, self).__init__()

        self.weights = nn.ParameterList()
        self.gcns = [GCNConv(1, 16, bias=False), GCNConv(16, 16, bias=False)]
        self.weights.append(self.gcns[0].weight)
        self.weights.append(self.gcns[1].weight)

        w = nn.Parameter(torch.ones(16, 11))
        nn.init.kaiming_normal_(w)
        self.weights.append(w)
        self.weights.append(nn.Parameter(torch.zeros(11)))

        self.tmpconv1 = GCNConv(1, 16, bias=False)
        self.tmpconv2 = GCNConv(16, 16, bias=False)

    def forward(self, data, vars=None):
        x, edge_idx = data.x, data.edge_index

        if vars is None:
            x = F.relu(self.gcns[0](x, edge_idx))
            x = F.relu(self.gcns[1](x, edge_idx))
            x = torch.mean(x, dim=0).reshape(-1, 16)
            x = torch.matmul(x, self.weights[2]).squeeze() + self.weights[3]

        else:
            self.tmpconv1.weight = Parameter(vars[0])
            self.tmpconv2.weight = Parameter(vars[1])

            x = F.relu(self.tmpconv1(x, edge_idx))
            x = F.relu(self.tmpconv2(x, edge_idx))
            x = torch.mean(x, dim=0).reshape(-1, 16)
            x = torch.matmul(x, vars[2]).squeeze() + vars[3]

        return x

    def assign_weight(self, weight):
        self.gcns[0].weight = Parameter(torch.clone(weight[0]))
        self.weights[0] = self.gcns[0].weight
        self.gcns[1].weight = Parameter(torch.clone(weight[1]))
        self.weights[1] = self.gcns[1].weight
        self.weights[2] = Parameter(torch.clone(weight[2]))
        self.weights[3] = Parameter(torch.clone(weight[3]))

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
