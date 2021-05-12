import torch
from torch import nn
from torch.nn import functional as F


class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, training,
                 num_features_nonzero=1,
                 dropout=0.,
                 bias=False,
                 activation=F.relu):
        super(GraphConvolutionLayer, self).__init__()
        self.training = training
        self.dropout = dropout
        self.activation = activation
        self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs):
        x, support = inputs

        if self.training:
            x = F.dropout(x, self.dropout)

        xw = torch.mm(x, self.weight)
        out = torch.sparse.mm(support, xw)

        if self.bias is not None:
            out += self.bias

        return self.activation(out)
