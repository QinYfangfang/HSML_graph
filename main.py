import torch
import numpy as np
import argparse
from torch.utils.data import random_split
from preprocess import Reddit12k, MetaLoader
from meta import Meta




def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Meta(args, device)
    dataset = Reddit12k(device)
    spt_ds, qry_ds = random_split(dataset, [int(0.8 * len(dataset))])
    data_loader = MetaLoader(dataset, args.k_spt, args.k_qry, batch_size=args.task_num)

    for step in range(args.epoch):
        x_spt, y_spt, x_qry, y_qry = data_loader.next()
        accs, loss = model(x_spt, y_spt, x_qry, y_qry)
        if step % 500 == 0:
            print('step:', step, '\ttraining acc:', accs, '\ttraining loss:', loss)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--epoch', type=int, help='epoch number', default=400)
    argparser.add_argument('--n_way', type=int, help='n way', default=11)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    arg_s = argparser.parse_args()
    main(arg_s)
