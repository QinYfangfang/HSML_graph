import torch_geometric.data
from torch_geometric.data import Data, DataLoader
from torch_geometric.data import InMemoryDataset
import torch
import numpy as np


class Reddit12k(InMemoryDataset):
    def __init__(self, device, label, root='./data', transform=None, pre_transform=None):
        self.label = label
        super(Reddit12k, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[label])
        self.data.to(device)

    @property
    def raw_file_names(self):
        return ['./data/Reddit12k/Reddit.txt']

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(11)]

    def download(self):
        pass

    def process(self):
        data_list = create_label_wise_data(label=self.label)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[self.label])


class MetaLoader:
    def __init__(self, dataset, k_spt, k_qry, batch_size=25, n_ways=11):
        self.k_spt = k_spt
        self.k_qry = k_qry
        self.n_ways = n_ways
        self.batch_size = batch_size
        self.loader = DataLoader(dataset, batch_size=k_spt+k_qry, shuffle=True)

    def next(self):
        x_spt, y_spt, x_qry, y_qry = [], [], [], []
        for i, data in enumerate(self.loader):
            data_spt = data[:self.k_spt]
            data_qry = data[self.k_spt:]
            x_spt += data_spt
            y_spt += [spt.y for spt in data_spt]
            x_qry += data_qry
            y_qry += [qry.y for qry in data_qry]
            if i+1 == self.batch_size:
                break

        return x_spt, y_spt, x_qry, y_qry


def create_label_wise_data(label, root='./data/Reddit12k/Reddit.txt'):
    graph_dict = {idx: [] for idx in range(11)}
    with open(root, 'r') as f:
        graph_num = int(f.readline())
        for _ in range(graph_num):
            line = f.readline().split()
            m, graph_class = int(line[0]), int(line[1])
            y = torch.zeros(1, 1, dtype=torch.long)
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
            graph_dict[y.item()].append(graph)
    return graph_dict[label]

def generate(path='./data/Reddit12k/Reddit.txt'):
    with open(path, 'r') as f:
        graph_num = int(f.readline())
        graph_list = []
        for _ in range(graph_num):
            line = f.readline().split()
            m, graph_class = int(line[0]), int(line[1])
            y = torch.zeros(1, 1, dtype=torch.long)
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


class BatchLoader:
    def __init__(self, datadict, k, batch_size=25, n_ways=11):
        self.k = k
        self.n_ways = n_ways
        self.batch_size = batch_size
        self.loader = {i: DataLoader(datadict[i], batch_size=k, shuffle=True) for i in datadict}
        self.iter = {i: iter(self.loader[i]) for i in self.loader}

    def next(self):
        ret = []

        for _ in range(self.batch_size):
            ret.append([])
            for i in self.iter:
                ret[_] += next(self.iter[i]).to_data_list()

        for i in range(self.batch_size):
            batch_ = torch_geometric.data.Batch()
            ret[i] = batch_.from_data_list(ret[i])

        return ret


