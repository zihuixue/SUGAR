import os
import time
import torch
import networkx as nx
import numpy as np
import metis


def agg_pred(fullgraph, sg_list, pred_list, evaluator_wrapper, datatype, detail=False):
    print(f'{datatype} data')
    labels = fullgraph.labels
    fullpred = torch.zeros((fullgraph.n_nodes, pred_list[0].shape[1]), dtype=pred_list[0].dtype).to(pred_list[0].device)
    predcnt = torch.zeros(fullgraph.n_nodes).to(pred_list[0].device)
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
        predcnt[sg.get_orig_id(idx)] = predcnt[sg.get_orig_id(idx)] + 1
        fullpred[sg.get_orig_id(idx)] = fullpred[sg.get_orig_id(idx)] + pred[idx]

    fullpred = (fullpred.T / predcnt).T

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


def normalize(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def sortbyfreq(arr):
    counts = {n: list(arr).count(n) for n in set(arr)}
    return sorted(list(set(arr)), key=lambda n: -counts[n])


def dgl2nxg(g):
    edges = g.edges()
    edgelist = []
    for i, j in zip(edges[0].numpy(), edges[1].numpy()):
        edgelist.append((i, j))
    g1 = nx.Graph()
    g1.add_edges_from(edgelist)
    return g1


def part_graph(g, dataset, part_num, part_mode, seed, tor, first_time, n_ratio=0.0):
    os.makedirs(os.path.join('./partition', dataset, str(part_num)), exist_ok=True)
    file_name1 = os.path.join('./partition', dataset, str(part_num), 'mode' + str(part_mode) + '-tor' + str(tor) + '.npy')
    file_name2 = os.path.join('./partition', dataset, str(part_num), 'nodelist.npy')

    if first_time:
        tic = time.perf_counter()
        if part_mode == 0:
            edges = g.edges()
            edgelist = []
            for i, j in zip(edges[0].numpy(), edges[1].numpy()):
                edgelist.append((i, j))
            g1 = nx.Graph()
            g1.add_edges_from(edgelist)

        else:
            print('Define edge weights according to degree')
            edges = g.edges()
            weight_np = g.in_degrees(edges[0]) + g.out_degrees(edges[0]) + g.in_degrees(edges[1]) + g.out_degrees(edges[1])
            weight_np = weight_np.numpy()
            if part_mode > 0:
                weight_np = np.max(weight_np) - weight_np + 1
            if abs(part_mode) > 1:
                weight_np = normalize(weight_np) * abs(part_mode)
            print('Edge weights', weight_np, 'Mean', np.mean(weight_np))

            edgelist = []
            idx = 0
            for i, j in zip(edges[0].numpy(), edges[1].numpy()):
                # tmp = g.in_degree(i) + g.out_degree(i) + g.in_degree(j) + g.out_degree(j)
                # assert tmp == weight_np[idx]
                edgelist.append((i, j, int(weight_np[idx]) + 1))
                idx = idx + 1

            g1 = nx.Graph()
            g1.add_weighted_edges_from(edgelist)
            g1.graph['edge_weight_attr'] = 'weight'
        print(f'Finish converting to networkx')

        min_edgecuts = 1e10
        for n_tor in range(tor):
            (edgecuts, parts) = metis.part_graph(g1, part_num, seed=seed + n_tor)
            if edgecuts < min_edgecuts:
                min_edgecuts = edgecuts
                best_parts = parts
            print(f'Round {n_tor}, edge cuts {edgecuts}, min edge cuts {min_edgecuts}')
        print(f'Finish metis, edge cuts = {min_edgecuts}, original graph # edge = {g1.number_of_edges()}')
        node_list = np.array([n for n in g1.nodes()])
        np.save(file_name1, best_parts)
        np.save(file_name2, node_list)
        print(f'Saving node list to {file_name1} and {file_name2}')

    else:
        best_parts = np.load(file_name1)
        node_list = np.load(file_name2)
        print(f'Loading node list from {file_name1} and {file_name2}')

    part_id_list = []
    edge_num_sum = 0
    if first_time and n_ratio > 0:
        assert g is not None
        g1 = dgl2nxg(g)
        for j in range(part_num):
            nodes_part = node_list[np.argwhere(np.array(best_parts) == j).ravel()]
            neighbor_list = []
            for i in nodes_part:
                for k in g1.neighbors(i):
                    if k not in nodes_part:
                        neighbor_list.append(k)
            neighbor_list_sorted = sortbyfreq(neighbor_list)
            print(f'Subgraph {j} # one-hop neighbor, before | after sorting {len(neighbor_list)} | {len(neighbor_list_sorted)}')
            node_num = int(n_ratio * len(neighbor_list_sorted))
            nodes_added = neighbor_list_sorted[0:node_num]
            nodes_part_new = np.concatenate((nodes_part, nodes_added))
            print(f'Adding {n_ratio*100} % nodes, num = {len(nodes_added)}, total num {len(nodes_part_new)}')
            part_id_list.append(nodes_part_new)
    else:
        for j in range(part_num):
            nodes_part = node_list[np.argwhere(np.array(best_parts) == j).ravel()]
            # sg0 = nx.subgraph(g1, nodes_part)
            # print(f'Partition {j}, node num = {sg0.number_of_nodes()}, edge num = {sg0.number_of_edges()}')
            # edge_num_sum = edge_num_sum + sg0.number_of_edges()
            part_id_list.append(nodes_part)
        # print(f'Keeping {edge_num_sum / g1.number_of_edges() * 100:.2f}% edges')
        # print('-' * 60)
    return part_id_list