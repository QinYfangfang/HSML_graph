import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np

from learner import Learner
from task_embedding import MEANAutoencoder
from lstm_tree import TreeLSTM
from graph_encoder import GraphEncoder


class Meta(nn.Module):
    def __init__(self, args, device):
        super(Meta, self).__init__()

        self.args = args
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.graph_embed = GraphEncoder().to(device)
        self.task_ae = MEANAutoencoder(input_size=64+11, hidden_num=args.hidden_dim).to(device)
        self.tree = TreeLSTM(device, args).to(device)
        self.net = Learner().to(device)

        self.task_mapping = []
        for weight in self.net.parameters():
            weight_size = np.prod(list(weight.size()))
            self.task_mapping.append(nn.Sequential(
                nn.Linear(2 * args.hidden_dim, weight_size.item()),
                nn.LeakyReLU()
            ).to(device))

        self.net_optim = optim.Adam(
            [{'params': self.tree.parameters()}, {'params': self.task_ae.parameters(), 'lr': 1e-4}]
            + [{'params': self.task_mapping[i].parameters() for i in range(len(self.task_mapping))}]
            + [{'params': self.graph_embed.parameters()}, {'params': self.net.parameters()}],
            lr=self.meta_lr)

        self.loss_q = torch.tensor(0.).to(device)
        self.loss_q.requires_grad_()

    def forward(self, spt, qry):
        task_lossa, task_lossesb, task_embed_loss = [], [], []
        task_outputa, task_outputbs = [], []
        accs_a, accs_b = 0., 0.
        task_weights, fast_weights = [], []

        for i in range(self.task_num):
            spt_graph_embed = self.graph_embed(spt[i].x, spt[i].edge_index, spt[i].batch)
            spt_one_hot = F.one_hot(spt[i].y.reshape(-1), num_classes=11)

            spt_graph_embed = torch.cat((spt_graph_embed, spt_one_hot), dim=-1)
            task_embed_vec, embed_loss = self.task_ae(spt_graph_embed)
            task_embed_loss.append(embed_loss)

            meta_knowledge_h = self.tree(task_embed_vec).reshape(1, -1)
            task_enhanced_emb_vec = torch.cat([task_embed_vec, meta_knowledge_h], dim=1)

            eta = [self.task_mapping[idx](task_enhanced_emb_vec).reshape(weight.size())
                   for idx, weight in enumerate(self.net.parameters())]

            task_weights.append([self.net.weights[idx] * eta[idx]
                                 for idx, weight in enumerate(self.net.parameters())])
            task_outputa.append(self.net(spt[i], task_weights[-1]))
            task_lossa.append(F.cross_entropy(task_outputa[-1].view(-1, 11), spt[i].y.view(-1)))

            with torch.no_grad():
                preda = F.softmax(task_outputa[-1], dim=1).argmax(dim=1)
                correct = torch.eq(preda, spt[i].y.view(-1)).sum().item()
                accs_a += correct / preda.size(0)

            grads = torch.autograd.grad(task_lossa, task_weights[-1])
            fast_weights.append(list(map(lambda p: p[1] - self.update_lr * p[0],
                                         zip(grads, task_weights[-1]))))
            task_outputbs.append(self.net(qry[i], fast_weights[-1]))
            task_lossesb.append(F.cross_entropy(task_outputbs[-1].view(-1, 11), qry[i].y.view(-1)))

            with torch.no_grad():
                predb = F.softmax(task_outputbs[-1], dim=1).argmax(dim=1)
                correct = torch.eq(predb, qry[i].y.view(-1)).sum().item()
                accs_b += correct / predb.size(0)

        tot_loss1 = sum(task_lossa) / self.task_num
        tot_loss2 = sum(task_lossesb) / self.task_num
        tot_embed_loss = sum(task_embed_loss) / self.task_num
        l1, l2, lb = tot_loss1.item(), tot_loss2.item(), tot_embed_loss.item()

        total_accuracy1 = accs_a / self.task_num
        total_accuracy2 = accs_b / self.task_num


        if self.args.metatrain_iterations > 0:
            gvs = tot_loss2 + self.args.emb_loss_weight * tot_embed_loss
            self.net_optim.zero_grad()
            gvs.backward()
            nn.utils.clip_grad_norm(self.parameters(), 5.)
            self.net_optim.step()

        return total_accuracy1, total_accuracy2, l1, l2, lb
