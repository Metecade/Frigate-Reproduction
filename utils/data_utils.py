import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import coo_array

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import NeighborSampler


class TrafficDataset(Dataset):
    def __init__(this, xs, ys, ls, mu, sig, unseen_node_ids, val=False):
        this.xs = xs
        this.ys = ys
        this.ls = ls # lipschitz
        this.mu = mu
        this.sig = sig
        this.unseen_node_ids = unseen_node_ids
        this.val = val
    def __len__(this):
        return len(this.xs)
    def __getitem__(this, idx):
        x = (this.xs[idx] - this.mu) / this.sig
        x[:, this.unseen_node_ids, :] = -this.mu / this.sig
        y = this.ys[idx]
        if not this.val:
            y[:, this.unseen_node_ids, :] = -this.mu / this.sig
        l = np.tile(this.ls.reshape(1, this.ls.shape[0], -1),
                (x.shape[0], 1, 1))
        x = np.concatenate((x, l), axis=2)
        return x, y


def get_xy(data, f, p):
    x_offsets = np.arange(-(p - 1), 1)
    y_offsets = np.arange(1, f + 1)
    # tmin - (p - 1) = 0 => tmin = p - 1
    # tmax + f = L - 1   => tmax = L - f - 1
    tmin = p - 1
    tmax = len(data) - f - 1
    xs, ys = [], []
    for t in range(tmin, tmax + 1): # tmax inclusive range
        xs.append(data[t + x_offsets, :, :])
        ys.append(data[t + y_offsets, :, :])
    xs, ys = list(map(np.stack, [xs, ys]))
    return xs, ys


def get_dataloader(traffic_path, lipschitz_path, keep_tod, f, p, unseen_node_ids):
    traffic_data_df = pd.read_pickle(traffic_path)
    traffic_data = traffic_data_df.values
    mu, sig = np.mean(traffic_data), np.std(traffic_data)
    if keep_tod:
        index = traffic_data_df.index.values
        index = ((index.astype('datetime64[ns]') - index.astype('datetime64[D]'))/
                np.timedelta64(1,'D'))
        tod = np.tile(index, (traffic_data.shape[1], 1)).T
        traffic_data = np.transpose(np.stack((traffic_data, tod)), (1, 2, 0))
    else:
        traffic_data = traffic_data.reshape(traffic_data.shape[0], -1, 1)
    cut_point1 = int(0.7 * len(traffic_data))
    cut_point2 = int(0.9 * len(traffic_data))
    train_data = traffic_data[:cut_point1, :, :]
    val_data = traffic_data[cut_point1:cut_point2, :, :]
    xys = list(map(lambda arg:get_xy(arg, f, p), [train_data, val_data]))
    ls = np.load(lipschitz_path)['lipschitz']
    train_datasets = TrafficDataset(*xys[0], ls, mu, sig, unseen_node_ids)
    val_datasets = TrafficDataset(*xys[1], ls, mu, sig, unseen_node_ids, val=True)
    train_loader = DataLoader(train_datasets, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_datasets, batch_size=32, shuffle=False)
    return train_loader, val_loader


def get_adjacency(adj_path):
    with open(adj_path, 'rb') as pkl:
        sparse_adj_data = pickle.load(pkl)
    v = sparse_adj_data['v']
    ij = sparse_adj_data['ij']
    shape = sparse_adj_data['shape']
    adj_mx = torch.tensor(coo_array((v, ij), shape=shape).todense())
    edge_index, edge_weight = dense_to_sparse(adj_mx)
    return edge_index, edge_weight, sparse_adj_data['shape']


def get_nbrloader(edge_index, node_ids, nlayers):
    node_ids = torch.tensor(node_ids, dtype=torch.long)
    return NeighborSampler(edge_index, node_idx=node_ids, batch_size=32,
            sizes=[-1 for _ in range(nlayers)])


def get_dataloader_and_adj_mx(traffic_path, lipschitz_path, adj_path, seen_path,
        *, keep_tod=True, f=12, p=12, nlayers=10):
    seen_node_ids = np.load(seen_path)
    edge_index, edge_weight, shape = get_adjacency(adj_path)
    unseen_node_ids = np.setdiff1d(np.arange(shape[0]), seen_node_ids)
    rev_index = torch.flip(edge_index, dims=[0])

    dataloaders = get_dataloader(traffic_path, lipschitz_path, keep_tod, f, p,
            unseen_node_ids)
    train_nbrloader = get_nbrloader(edge_index, seen_node_ids, nlayers)
    rev_train_nbrloader = get_nbrloader(rev_index, seen_node_ids, nlayers)
    val_nbrloader = get_nbrloader(edge_index, unseen_node_ids, nlayers)
    rev_val_nbrloader = get_nbrloader(rev_index, unseen_node_ids, nlayers)
    dataloaders_ = {'train_loader': {
                        'dataloader': dataloaders[0],
                        'neighbor_loader': train_nbrloader,
                        'rev_loader': rev_train_nbrloader,
                        },
                     'val_loader': {
                         'dataloader': dataloaders[1],
                         'neighbor_loader': val_nbrloader,
                         'rev_loader': rev_val_nbrloader,
                        },
                    }
    return dataloaders_, edge_weight
