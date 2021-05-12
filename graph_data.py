import numpy as np
import torch
from torch_geometric.data import Data, Dataset


def normalize(A):
    A += torch.eye(A.size(0))
    d = A.sum(1)
    D = torch.diag(torch.pow(d, -0.5))
    return D.mm(A).mm(D)




'''
class GraphData(Dataset):
    def __init__(self, path='./data/Reddit12k/Reddit.txt'):
        self.path = path
        self.num_classes = 11
        self.graphs = []
        with open(path, 'r') as f:
            graph_num = int(f.readline())
            for _ in range(graph_num):
                line = f.readline().split()
                m, graph_class = int(line[0]), int(line[1])
                y = graph_class-1
                x = np.ones(shape=(m, 1))
                adj = torch.ones(m, m)
                for node in range(m):
                    line = [int(xx) for xx in f.readline().split()]
                    x[node, 0] = line[1]
                    for nei in range(line[1]):
                        adj[node, nei] = 1
                adj = normalize(adj)
                graph = (x, y, adj)
                self.graphs.append(graph)

    def __getitem__(self, idx):
        assert idx < len(self.graphs)
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)
'''

def graph_data_loader(batch_size):
    return DataLoader(GraphData(), batch_size=batch_size, shuffle=True)




