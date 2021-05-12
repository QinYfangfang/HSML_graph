import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
from copy import deepcopy

from graph_encode.main import InfoGraph
from learner import Learner


class Meta(nn.Module):
    def __init__(self, args, device):
        super(Meta, self).__init__()

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

        self.net = Learner().to(device)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.loss_q = torch.tensor(0.).to(device)
        self.loss_q.requires_grad_()

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        corrects = 0.
        self.loss_q = 0.
        origin_weight = [torch.clone(p) for p in self.net.parameters()]

        for i in range(self.task_num):
            self.net.assign_weight(origin_weight)
            logits = self.net(x_spt[i]).reshape(1, -1)
            loss = F.cross_entropy(logits, y_spt[i][0])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            self.net.assign_weight(fast_weights)
            logits_q = self.net(x_qry[i]).reshape(1, -1)
            loss = F.cross_entropy(logits_q, y_qry[i][0])
            self.loss_q = self.loss_q + loss

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects = corrects + correct

        self.loss_q = self.loss_q / self.task_num
        self.meta_optim.zero_grad()
        self.loss_q.backward(retain_graph=True)
        '''
        print('meta update')
        for p in self.net.parameters()[:4]:
            print(torch.norm(p).item())
        '''
        self.meta_optim.step()
        accs = corrects / (self.task_num * x_qry[0].y.size(0))
        return accs, self.loss_q.item()

    def finetuning(self, x_spt, y_spt, x_qry, y_qry):
        querysz = x_qry.size(0)
        corrects = [0 for _ in range(self.update_step_test + 1)]
        net = deepcopy(self.net)

        logits = self.net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, self.net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

        with torch.no_grad():
            logits_q = self.net(x_qry, fast_weights)
            loss = F.cross_entropy(logits_q, y_qry)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects = corrects + correct

        del net
        accs = np.array(corrects) / querysz

        return accs
