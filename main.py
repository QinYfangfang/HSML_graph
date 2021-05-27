import torch
import numpy as np
import argparse
from torch.utils.data import random_split
from preprocess import Reddit12k, MetaLoader, BatchLoader
from meta import Meta
from torch.optim import lr_scheduler


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Meta(args, device)
    dataset = {i: Reddit12k(device, i) for i in range(11)}
    reses = ['epoch', 'acc_tr', 'acc_te', 'loss_tr', 'loss_te', 'embed_loss']
    res_dict = {st: [] for st in reses}
    scheduler = lr_scheduler.ExponentialLR(model.net_optim, gamma=0.99)
    print(f'Device: {device}, start traing.')
    train_classes = [1, 3, 5, 6, 7, 9, 11]
    test_classes = [2, 4, 8, 10]
    for step in range(args.epoch):
        spt_loader = BatchLoader(dataset, args.k_spt, train_classes, batch_size=args.task_num)
        qry_loader = BatchLoader(dataset, args.k_qry, train_classes, batch_size=args.task_num)
        epoch_res = []
        while True:
            try:
                spt_batch = spt_loader.next()
                qry_batch = qry_loader.next()
                res = model(spt_batch, qry_batch)
                epoch_res.append(res)
            except StopIteration:
                break
        if step <= 1000:
            scheduler.step()
        if step % 50 == 0:
            res_dict['epoch'].append(step)
            for idx, key in enumerate(res_dict.keys()):
                if not idx:
                    continue
                res_dict[key].append(0.)
                for item in epoch_res:
                    res_dict[key][-1] += item[idx-1]
                res_dict[key][-1] /= len(epoch_res)
            print(f'Epoch {step}:')
            print('\tTrain_acc: {}, Test_acc: {}'.format(res_dict['acc_tr'][-1], res_dict['acc_te'][-1]))
            print('\tTrain_loss: {}, Test_loss: {}, Embed_loss: {}'.format(
                res_dict['loss_tr'][-1], res_dict['loss_te'][-1], res_dict['embed_loss'][-1]))
        if (step + 1) % 100 == 0:
            state = {'net': model.state_dict(), 'optim': model.net_optim.state_dict(),
                     'scheduler': scheduler.state_dict(), 'epoch': step}
            dire = './saved_model/' + f'epoch_{step}'
            torch.save(state, dire)
            print(f'Model saved at: {dire}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--epoch', type=int, help='epoch number', default=4000)
    argparser.add_argument('--n_way', type=int, help='n way', default=11)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=10)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-2)
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
