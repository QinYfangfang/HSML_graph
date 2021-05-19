import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import optim
from torch.nn import Parameter
import numpy as np
from copy import deepcopy

from graph_encode.main import InfoGraph
from learner import Learner
from task_embedding import MEANAutoencoder
from lstm_tree import TreeLSTM


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

        self.graph_encoder = InfoGraph()
        self.graph_encoder.load_state_dict(torch.load('./graph_encode/Encoder_state_dict.pkl'))
        self.graph_encoder.to(device)

        self.task_ae = MEANAutoencoder(input_size=96+11, hidden_num=args.hidden_dim)
        self.task_ae.to(device)

        self.tree = TreeLSTM(device, args)
        self.tree.to(device)

        self.net = Learner().to(device)

        self.task_mapping = []
        for weight in self.net.parameters():
            weight_size = np.prod(list(weight.size()))
            self.task_mapping.append(nn.Sequential(
                nn.Linear(2 * args.hidden_dim, weight_size.item()),
                nn.Sigmoid()
            ).to(device))

        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.net_optim = optim.Adam(
            [{'params': self.tree.parameters()}]
            + [{'params': self.task_mapping[i].parameters() for i in range(len(self.task_mapping))}],
            lr=self.meta_lr)
        self.ae_potim = optim.SGD(self.task_ae.parameters(), lr=self.meta_lr)

        self.loss_q = torch.tensor(0.).to(device)
        self.loss_q.requires_grad_()


    def forward(self, spt, qry):
        task_lossa, task_lossesb, task_embed_loss = [], [], []
        task_outputa, task_outputbs = [], []
        accs_a, accs_b = 0., 0.

        for i in range(self.task_num):
            spt_graph_embed, _ = self.graph_encoder.encoder(spt[i].x, spt[i].edge_index, spt[i].batch)
            spt_one_hot = F.one_hot(spt[i].y.reshape(-1), num_classes=11)
            spt_graph_embed = torch.cat((spt_graph_embed, spt_one_hot), dim=-1)
            task_embed_vec, embed_loss = self.task_ae(spt_graph_embed)
            task_embed_loss.append(embed_loss)

            meta_knowledge_h = self.tree(task_embed_vec)
            task_enhanced_emb_vec = torch.cat([task_embed_vec, meta_knowledge_h], dim=1)

            eta = [self.task_mapping[idx](task_enhanced_emb_vec).reshape(weight.size())
                   for idx, weight in enumerate(self.net.parameters())]

            task_weights = [self.net.weights[idx] * eta[idx] for idx, weight in enumerate(self.net.parameters())]
            task_outputa.append(self.net(spt[i], task_weights))
            task_lossa.append(F.cross_entropy(task_outputa[-1].view(-1, 11), spt[i].y.view(-1)))

            with torch.no_grad():
                preda = F.softmax(task_outputa[-1], dim=1).argmax(dim=1)
                correct = torch.eq(preda, spt[i].y.view(-1)).sum().item()
                accs_a += correct / preda.size(0)

            grads = torch.autograd.grad(task_lossa, task_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grads, task_weights)))
            task_outputbs.append(self.net(qry[i], fast_weights))
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
            # gvs = tot_loss2 + self.args.emb_loss_weight * tot_embed_loss
            self.net_optim.zero_grad()
            tot_loss2.backward(retain_graph=True)
            '''
            for p in self.task_ae.parameters():
                print(p.grad)
            print('\n\n')
            raise RuntimeError
            '''
            self.net_optim.step()

            self.ae_potim.zero_grad()
            tot_embed_loss.backward()
            nn.utils.clip_grad_norm(self.task_ae.parameters(), 2.)

            self.ae_potim.step()

        return total_accuracy1, total_accuracy2, l1, l2, lb
