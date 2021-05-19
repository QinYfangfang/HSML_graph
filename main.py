import torch
import numpy as np
import argparse
from torch.utils.data import random_split
from preprocess import Reddit12k, MetaLoader, BatchLoader
from meta import Meta


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Meta(args, device)
    dataset = {i: Reddit12k(device, i) for i in range(11)}

    for step in range(args.epoch):
        spt_loader = BatchLoader(dataset, args.k_spt, batch_size=args.task_num)
        qry_loader = BatchLoader(dataset, args.k_qry, batch_size=args.task_num)
        while True:
            try:
                spt_batch = spt_loader.next()
                qry_batch = qry_loader.next()
                acc_tr, acc_te, loss_tr, loss_te, embed_loss = model(spt_batch, qry_batch)
            except StopIteration:
                break
        if step % 50 == 0:
            print(f'Step: {step}, acc_tr: {acc_tr}, acc_te: {acc_te},' +
                  f' loss_tr: {loss_tr}, loss_te: {loss_te}, embed_loss: {embed_loss}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--epoch', type=int, help='epoch number', default=4000)
    argparser.add_argument('--n_way', type=int, help='n way', default=11)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-4)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--hidden_dim', type=int, help='hidden dim for task autoencoder', default=40)
    argparser.add_argument('--cluster_layer_0', type=int, help='num of leaves in cluster layer 0', default=4)
    argparser.add_argument('--cluster_layer_1', type=int, help='num of leaves in cluster layer 1', default=2)
    argparser.add_argument('--metatrain_iterations', type=int, help='number of metatraining iterations', default=15000)
    argparser.add_argument('--emb_loss_weight', type=float, help='number of metatraining iterations', default=0.5)

    arg_s = argparser.parse_args()
    main(arg_s)
