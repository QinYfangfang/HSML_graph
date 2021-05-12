from torch_geometric.data import Data, DataLoader
from torch_geometric.data import InMemoryDataset
import torch


class Reddit12k(InMemoryDataset):
    def __init__(self, root='./data', transform=None, pre_transform=None):
        super(Reddit12k, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['./data/Reddit12k/Reddit.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = generate()
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def generate(path='./data/Reddit12k/Reddit.txt'):
    with open(path, 'r') as f:
        graph_num = int(f.readline())
        graph_list = []
        for _ in range(graph_num):
            line = f.readline().split()
            m, graph_class = int(line[0]), int(line[1])
            y = torch.zeros(1, 1)
            y[0, 0] = graph_class - 1
            x = torch.ones(m, 1)
            adj = [[], []]
            for node in range(m):
                line = [int(xx) for xx in f.readline().split()]
                x[node, 0] = line[1]
                for nei in range(line[1]):
                    adj[0].append(node)
                    adj[0].append(nei)
                    adj[1].append(nei)
                    adj[1].append(node)
            adj = torch.LongTensor(adj)
            graph = Data(x=x, edge_index=adj, y=y)
            graph_list.append(graph)
    return graph_list

