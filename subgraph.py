import os
import time
import metis
import networkx as nx
import numpy as np
import random
import codecs
import json
import argparse
import torch
import dgl
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset


import sys, os
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from part_utils import part_graph


def map_idx(idx, orig_id):
    inter_idx = np.intersect1d(idx, orig_id)
    newid = []
    for i in inter_idx:
        tmp = np.where(orig_id == i)[0]
        assert len(tmp) == 1
        newid.append(tmp[0])
    return np.array(newid)


def gen_mask(n_nodes, idx):
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    mask[idx] = True
    return mask

def map_mask(mask):
    tmp = torch.where(mask == True)
    return tmp[0]


class FullGraph:
    def __init__(self, dataset, device, self_loop=True):
        data = DglNodePropPredDataset(dataset)
        graph, labels = data[0]
        self.gid = -1
        self.dataset = dataset
        self.graph = self.preprocess(graph, self_loop).to(device)
        self.labels = labels.to(device)
        self.n_nodes = self.graph.number_of_nodes()
        splitted_idx = data.get_idx_split()
        self.train_idx, self.val_idx, self.test_idx = splitted_idx["train"].to(device), splitted_idx["valid"].to(device), splitted_idx["test"].to(device)
        self.in_feats = graph.ndata['feat'].shape[1]
        if dataset == 'ogbn-proteins':
            self.n_classes = 112
        else:
            self.n_classes = (labels.max() + 1).item()

    @staticmethod
    def preprocess(graph, self_loop):
        graph = graph.remove_self_loop().to_simple()
        if self_loop:
            srcs, dsts = graph.all_edges()
            graph.add_edges(dsts, srcs)
            graph = graph.remove_self_loop().add_self_loop()
        # graph.create_formats_()
        return graph

    def rem_edge(self, nodes_list):

        edge_ids = torch.IntTensor([])
        for n in nodes_list:
            sg = dgl.node_subgraph(self.graph, n)
            eid = sg.edata['_ID']
            edge_ids = torch.cat((edge_ids, eid))

        fg_eid = self.graph.edges('eid').tolist()
        sg_eid = edge_ids.tolist()
        rem_eid = list(set(fg_eid).difference(set(sg_eid)))
        self.graph.remove_edges(rem_eid)
        print(f'Removing {len(rem_eid)} from {len(fg_eid)} edges, ratio {len(rem_eid) / len(fg_eid) * 100:.2f}%')


class SubGraph:
    def __init__(self, dataset, gid, num_parts, part_mode, nodes_id, device, first_time, n_ratio):
        self.gid = gid   # subgraph id
        self.device = device
        self.graph, self.labels, self.train_idx, self.val_idx, self.test_idx \
            = self.load_subgraph(dataset, num_parts, part_mode, nodes_id, first_time, n_ratio)
        self.orig_id = self.graph.ndata['orig_id']
        self.in_feats = self.graph.ndata['feat'].shape[1]
        self.n_classes = 40

    def get_orig_id(self, idx):
        return self.graph.ndata['orig_id'][idx]

    def get_cur_id(self, idx):
        cur_idx = (self.orig_id == idx).nonzero(as_tuple=True)[0][0]
        return cur_idx

    def load_subgraph(self, dataset, num_parts, part_mode, nodes_id, first_time, n_ratio):
        fp = os.path.join('./partition', dataset, str(num_parts))
        os.makedirs(fp, exist_ok=True)
        if n_ratio > 0:
            fn = os.path.join(fp, 'mode' + str(part_mode) + 'ratio' + str(n_ratio) + '-sg' + str(self.gid) + '.bin')
        else:
            fn = os.path.join(fp, 'mode' + str(part_mode) + '-sg' + str(self.gid) + '.bin')
        if first_time:
            fg = FullGraph(dataset, 'cpu')
            g0 = dgl.node_subgraph(fg.graph, nodes_id)

            # set node features
            g0_orig_id = g0.ndata['_ID']
            if dataset == 'ogbn-arxiv':
                # add reverse edges
                srcs, dsts = g0.all_edges()
                print(f"Total edges before adding reverse edges {g0.number_of_edges()}")
                g0.add_edges(dsts, srcs)
                print(f"Total edges after adding reverse edges {g0.number_of_edges()}")

            g0.ndata['orig_id'] = g0_orig_id
            g0.ndata["feat"] = fg.graph.ndata['feat'][g0_orig_id]
            g0.ndata['labels'] = fg.labels[g0_orig_id]

            # add self-loop
            print(f"Total edges before adding self-loop {g0.number_of_edges()}")
            g0 = g0.remove_self_loop().add_self_loop()
            print(f"Total edges after adding self-loop {g0.number_of_edges()}")

            print(f"full graph: node number {fg.graph.number_of_nodes()}, edge number {fg.graph.number_of_edges()}")
            print(f"subgraph: node number {g0.number_of_nodes()}, edge number {g0.number_of_edges()}")

            # map train, val and test idx
            g0_orig_id = g0.ndata['orig_id'].numpy()
            g0_labels = fg.labels[g0_orig_id]
            g0.ndata['labels'] = g0_labels

            g0_train_idx = map_idx(fg.train_idx, g0_orig_id)
            g0_val_idx = map_idx(fg.val_idx, g0_orig_id)
            g0_test_idx = map_idx(fg.test_idx, g0_orig_id)

            g0.ndata['train_mask'] = gen_mask(g0.number_of_nodes(), g0_train_idx)
            g0.ndata['val_mask'] = gen_mask(g0.number_of_nodes(), g0_val_idx)
            g0.ndata['test_mask'] = gen_mask(g0.number_of_nodes(), g0_test_idx)

            print(f'saving file to {fn}')
            dgl.save_graphs(fn, g0)

            g0_train_idx, g0_val_idx, g0_test_idx = map(lambda x: torch.from_numpy(x).long(),
                                                        (g0_train_idx, g0_val_idx, g0_test_idx))

        else:
            print(f'loading file from {fn}')
            data = dgl.load_graphs(fn)
            g0 = data[0][0]
            g0_labels = g0.ndata['labels']
            g0_train_idx = map_mask(g0.ndata['train_mask'])
            g0_val_idx = map_mask(g0.ndata['val_mask'])
            g0_test_idx = map_mask(g0.ndata['test_mask'])

        g0, g0_labels, g0_train_idx, g0_val_idx, g0_test_idx = map(
            lambda x: x.to(self.device), (g0, g0_labels, g0_train_idx, g0_val_idx, g0_test_idx)
        )

        print(f"Subgraph: node number {g0.number_of_nodes()}, edge number {g0.number_of_edges()}")
        print(f"Finish loading subgraph {self.gid}")
        print('-' * 70)

        return g0, g0_labels, g0_train_idx, g0_val_idx, g0_test_idx


def load_subgraph(fg, args, seed):
    g = fg.graph if fg is not None else None
    nodes_list = part_graph(g, args.dataset, args.num_parts, args.part_mode, seed, args.num_tor, args.first_time, args.n_ratio)
    # sg_num_list = list(map(int, args.partition.split(',')))
    sg_list = []
    for idx, nodes_id in enumerate(nodes_list):
        # if idx in sg_num_list:
        sg = SubGraph(args.dataset, idx, args.num_parts, args.part_mode, nodes_id, 'cpu', args.first_time, args.n_ratio)
        sg_list.append(sg)
    return sg_list


def run():
    argparser = argparse.ArgumentParser(description='preprocess')
    argparser.add_argument("--gpu", type=int, default=-1, help="GPU device ID.")
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument("--dataset", type=str, default="ogbn-arxiv", help="dataset name")
    argparser.add_argument("--first-time", action='store_true', help="run metis for first time")
    argparser.add_argument("--num-tor", type=int, default=5, help="num of metis partitions")
    argparser.add_argument("--num-parts", type=int, default=2, help="number of partitions")
    argparser.add_argument("--n-ratio", type=float, default=0.0, help="redundant node ratio")
    argparser.add_argument("--partition", type=str, default="0,1", help="run the model on which subgraph")
    argparser.add_argument("--part-mode", type=int, default=0, help="how to partition the graph")
    args = argparser.parse_args()
    print(args)

    device = torch.device("cpu") if args.gpu < 0 else torch.device("cuda:" + str(args.gpu))
    args.device = device

    # Load graph
    fg = FullGraph(args.dataset, 'cpu', False)
    sg_list = load_subgraph(fg, args, args.seed)


if __name__ == '__main__':
    run()