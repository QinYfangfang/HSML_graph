import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class TreeLSTM(nn.Module):
    def __init__(self, device, args):
        super(TreeLSTM, self).__init__()
        self.input_dim = args.hidden_dim
        self.tree_hidden_dim = args.hidden_dim
        self.cluster_layer_0 = args.cluster_layer_0
        self.cluster_layer_1 = args.cluster_layer_1

        self.leaf_i, self.leaf_o, self.leaf_u = [], [], []
        for i in range(args.cluster_layer_0):
            self.leaf_u.append(nn.Linear(self.input_dim, self.tree_hidden_dim).to(device))

        self.no_leaf_i, self.no_leaf_o, self.no_leaf_u = [], [], []
        for i in range(args.cluster_layer_1):
            self.no_leaf_i.append(nn.Linear(self.tree_hidden_dim, 1).to(device))
            self.no_leaf_u.append(nn.Linear(self.tree_hidden_dim, self.tree_hidden_dim).to(device))

        self.root_u = nn.Linear(self.tree_hidden_dim, self.tree_hidden_dim)
        self.cluster_center = []

        for i in range(args.cluster_layer_0):
            self.cluster_center.append(Variable(torch.randn(size=(1, self.input_dim))).to(device))

    def forward(self, inputs):
        sigma = 2.0
        for idx in range(self.cluster_layer_0):
            if idx == 0:
                all_value = torch.exp(
                    -torch.sum(torch.square(inputs - self.cluster_center[idx])) / (2. * sigma)
                )
            else:
                all_value = all_value + torch.exp(
                    -torch.sum(torch.square(inputs - self.cluster_center[idx])) / (2. * sigma)
                )

        c_leaf = []
        for idx in range(self.cluster_layer_0):
            assignment_idx = torch.exp(
                -torch.sum(torch.square(inputs - self.cluster_center[idx])) / (2. * sigma) / all_value
            )
            value_u = torch.tanh(self.leaf_u[idx](inputs))
            value_c = assignment_idx * value_u
            c_leaf.append(value_c)

        c_no_leaf = []
        for idx in range(self.cluster_layer_0):
            input_gate = []
            for idx_layer_1 in range(self.cluster_layer_1):
                input_gate.append(self.no_leaf_i[idx_layer_1](c_leaf[idx]))

            input_gate = F.softmax(torch.cat(input_gate, dim=0), dim=0)

            c_no_leaf_temp = []
            for idx_layer_1 in range(self.cluster_layer_1):
                no_leaf_value_u = torch.tanh(self.no_leaf_u[idx_layer_1](c_leaf[idx]))
                c_no_leaf_temp.append(input_gate[idx_layer_1] * no_leaf_value_u)

            c_no_leaf.append(torch.cat(c_no_leaf_temp, dim=0))

        c_no_leaf = torch.stack(c_no_leaf, dim=0)
        c_no_leaf = torch.transpose(c_no_leaf, 0, 1)
        c_no_leaf = torch.sum(c_no_leaf, dim=1, keepdim=True)

        root_c = []
        for idx in range(self.cluster_layer_1):
            root_c.append(torch.tanh(self.root_u(c_no_leaf)))

        root_c = torch.sum(torch.cat(root_c, dim=0), dim=0)
        return root_c
