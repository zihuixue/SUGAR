import copy
import os
import time
import argparse
import random
import math
import dgl
import psutil
import numpy as np
import torch
import torch.optim as optim

# import sys
# sys.path.insert(0, './')
from train_baseline import adjust_learning_rate, add_labels, train, cross_entropy
from subgraph import load_subgraph, FullGraph
from model_utils import gen_gcn_model, load_evaluator
from part_utils import agg_pred


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


@torch.no_grad()
def evaluate(model, graph, labels, train_idx, val_idx, test_idx, use_labels):
    model.eval()

    feat = graph.ndata["feat"]

    if use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat)
    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    return (
        evaluator_wrapper(pred[train_idx], labels[train_idx]),
        evaluator_wrapper(pred[val_idx], labels[val_idx]),
        evaluator_wrapper(pred[test_idx], labels[test_idx]),
        train_loss,
        val_loss,
        test_loss,
        pred
    )


def run_uni(args, sg):
    model = gen_gcn_model(args, args.n_layers, sg.in_feats, sg.n_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=1e-3)

    g = sg.graph.to(device)
    labels = sg.labels.to(device)

    total_time = 0
    best_val_acc, best_test_acc = 0, 0
    final_pred = None

    for epoch in range(args.n_epochs):
        if not cuda_disable:
            torch.cuda.synchronize()
        tic = time.perf_counter()

        adjust_learning_rate(optimizer, args.lr, epoch)
        loss, pred = train(model, g, labels, sg.train_idx, optimizer, args.use_labels)

        if not cuda_disable:
            torch.cuda.synchronize()
        toc = time.perf_counter()

        if epoch >= 5:
            total_time += toc - tic

        lr_scheduler.step(loss)

        if epoch % args.val_every == 0 and epoch != 0:
            train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = evaluate(
                model, g, labels, sg.train_idx, sg.val_idx, sg.test_idx, args.use_labels
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                final_pred = pred

        if epoch % args.log_every == 0 or epoch == args.n_epochs - 1:
            print(f"Graph id {sg.gid} | Epoch {epoch} | Best val {best_val_acc:.4f} | Best test acc: {best_test_acc:.4f}")
            print("-" * 70)

    print(f'Finish training, avg epoch time = {total_time / (epoch - 4):.5f}s')

    return best_val_acc, best_test_acc, final_pred


def run(args):
    if args.run_full:
        val_acc, test_acc, _ = run_uni(args, fg)
        return val_acc, test_acc

    pred_list = []
    for sg in sg_list:
        _, _, pred = run_uni(args, sg)
        pred_list.append(pred)

    train_acc = agg_pred(fg, sg_list, pred_list, evaluator_wrapper, 'train')
    val_acc = agg_pred(fg, sg_list, pred_list, evaluator_wrapper, 'val')
    test_acc = agg_pred(fg, sg_list, pred_list, evaluator_wrapper, 'test')

    print('Subgraph 0 / Subgraph 1 / ... / Full graph ')
    print(f'train score:', [round(i, 4) for i in train_acc])
    print(f'val score:', [round(i, 4) for i in val_acc])
    print(f'test score:', [round(i, 4) for i in test_acc])

    return val_acc[-1], test_acc[-1]


if __name__ == '__main__':
    global device, fg, sg_list, evaluator_wrapper

    argparser = argparse.ArgumentParser("GCN on OGBN-Arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--dataset", type=str, default="ogbn-arxiv", help="dataset name")
    argparser.add_argument("--gpu", type=int, default=1, help="GPU device ID.")
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument("--n-runs", type=int, default=1, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=1000, help="number of epochs")
    argparser.add_argument("--use-labels", action="store_true", help="Use labels in the training set as input features.")
    argparser.add_argument("--use-linear", action="store_true", help="Use linear layer.")
    argparser.add_argument('--save-pred', action="store_true", help="whether to save prediction")
    argparser.add_argument('--save-model', action="store_true", help="whether to save model")
    argparser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=3, help="number of layers")
    argparser.add_argument("--n-hidden", type=int, default=256, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--log-every", type=int, default=100, help="log every LOG_EVERY epochs")
    argparser.add_argument('--val-every', type=int, default=1)

    argparser.add_argument("--first-time", action='store_true', help="run metis for first time")
    argparser.add_argument("--run-full", action="store_true", help="run the full graph")
    argparser.add_argument("--num-tor", type=int, default=1, help="num of metis partitions")
    argparser.add_argument("--num-parts", type=int, default=2, help="num of partitions")
    argparser.add_argument("--n-ratio", type=float, default=0.0, help="redundant node ratio")
    argparser.add_argument("--partition", type=str, default="0,1", help="run the model on which subgraph")
    argparser.add_argument("--part-mode", type=int, default=100, help="how to partition the graph")
    args = argparser.parse_args()

    device = torch.device("cpu") if args.gpu < 0 else torch.device("cuda:" + str(args.gpu))
    cuda_disable = True if args.gpu < 0 else False
    args.device = device

    # Load graph
    if args.run_full:
        fg = FullGraph(args.dataset, 'cpu')
    else:
        fg = FullGraph(args.dataset, 'cpu', self_loop=False)
        sg_list = load_subgraph(fg, args, args.seed)

    # Load evaluator
    evaluator_wrapper = load_evaluator(args.dataset)

    val_accs, test_accs = [], []
    for i in range(args.n_runs):
        seed(args.seed + i)
        val_acc, test_acc = run(args)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    print(f"Runned {args.n_runs} times")
    print(f"Average val accuracy: {np.mean(val_accs, axis=0)} ± {np.std(val_accs, axis=0)}")
    print(f"Average test accuracy: {np.mean(test_accs, axis=0)} ± {np.std(test_accs, axis=0)}")
    print(args)
    print('Finish')