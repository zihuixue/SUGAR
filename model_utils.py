import os
import numpy as np
import codecs
import csv

import torch
import torch.nn.functional as F

from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from models import GCN


def gen_gcn_model(args, n_layers, in_feats, n_classes):
    if args.use_labels:
        model = GCN(in_feats + n_classes, args.n_hidden, n_classes, n_layers, F.relu, args.dropout, args.use_linear)
    else:
        model = GCN(in_feats, args.n_hidden, n_classes, n_layers, F.relu, args.dropout, args.use_linear)
    print('Num of params:', sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad]))

    return model


def load_evaluator(dataset):
    evaluator = Evaluator(name=dataset)
    if dataset == 'ogbn-proteins':
        evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred, "y_true": labels})["rocauc"]
    else:  # arxiv and products
        evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]
    return evaluator_wrapper


def agg_pred(fullgraph, sg_list, pred_list, evaluator_wrapper, datatype, detail=False):
    print(f'{datatype} data')
    labels = fullgraph.labels
    fullpred = torch.zeros((fullgraph.n_nodes, pred_list[0].shape[1]), dtype=pred_list[0].dtype).to(pred_list[0].device)
    sg_acc = []
    for i in range(len(sg_list)):
        sg = sg_list[i]
        pred = pred_list[i]
        if datatype == 'train':
            idx = sg.train_idx
        elif datatype == 'val':
            idx = sg.val_idx
        else:
            idx = sg.test_idx
        acc = evaluator_wrapper(pred[idx], sg.labels[idx])
        if detail:
            inner_idx = np.intersect1d(idx.cpu().numpy(), sg.inner_node_idx().cpu().numpy())
            in_acc = evaluator_wrapper(pred[inner_idx], sg.labels[inner_idx])
            halo_idx = np.intersect1d(idx.cpu().numpy(), sg.halo_node_idx().cpu().numpy())
            halo_acc = evaluator_wrapper(pred[halo_idx], sg.labels[halo_idx])
            print(f'Subgraph {i} accuracy: {acc*100:.2f}, Inner nodes acc {in_acc*100:.2f}, Halo nodes acc {halo_acc*100:.2f}')
        sg_acc.append(acc)
        # print('Accuracy on subgraph %i: %.3f' % (i, acc))
        # todo: different for ogbn-proteins when nodes overlap
        # fullpred[sg.get_orig_id(idx)] = pred[idx]
        fullpred[sg.get_orig_id(idx)] = fullpred[sg.get_orig_id(idx)] + pred[idx]

    if datatype == 'train':
        fullidx = fullgraph.train_idx
    elif datatype == 'val':
        fullidx = fullgraph.val_idx
    else:
        fullidx = fullgraph.test_idx
    print('-' * 70)
    acc = evaluator_wrapper(fullpred[fullidx], labels[fullidx])
    sg_acc.append(acc)
    # print('Aggregated accuracy: %.3f' % acc)
    return sg_acc


def save_pred(args, sg_id, final_pred):
    fp = os.path.join("./pred/", args.dataset, str(args.num_parts), str(args.n_ratio))
    os.makedirs(fp, exist_ok=True)
    print(f'Saving model {sg_id} to {fp}...')
    torch.save(final_pred.cpu(), os.path.join(fp, 'pred_m' + str(sg_id) + '_rand_' + str(args.rand)))


def save_model(args, sg_id, best_model):
    fp = os.path.join("./saved_models", args.dataset, str(args.num_parts))
    if args.rand_part:
        fp = fp + "rand"
    os.makedirs(fp, exist_ok=True)
    print(f'Saving model {sg_id} to {fp}...')
    torch.save(best_model.state_dict(), os.path.join(fp, "model_m" + str(sg_id) + '_nratio_' + str(args.n_ratio) + ".pth"))




