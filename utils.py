# coding=utf-8
"""
@author: Yantong Lai
@description: Common utilization for POI recommendation
"""

import pickle
import numpy as np
from math import radians, cos, sin, asin, sqrt
import scipy.sparse as sp
import torch


def get_unique_seq(sessions_list):
    """Get unique POIs in the sequence"""
    seq_list = []
    for session in sessions_list:
        for poi in session:
            if poi in seq_list:
                continue
            else:
                seq_list.append(poi)

    return seq_list


def get_unique_seqs_for_sessions(sessions_dict):
    """Get unique seq for each session"""
    seqs_dict = {}
    seqs_lens_dict = {}
    for key, value in sessions_dict.items():
        seqs_dict[key] = get_unique_seq(value)
        seqs_lens_dict[key] = len(get_unique_seq(value))

    return seqs_dict, seqs_lens_dict


def get_seqs_for_sessions(sessions_dict, padding_idx, max_seq_len):
    seqs_dict = {}
    seqs_lens_dict = {}
    reverse_seqs_dict = {}
    for key, sessions in sessions_dict.items():
        temp = []
        for session in sessions:
            temp.extend(session)
        if len(temp) >= max_seq_len:
            temp = temp[-max_seq_len:]
            temp_rev = temp[::-1]
            seqs_dict[key] = temp
            reverse_seqs_dict[key] = temp_rev
            seqs_lens_dict[key] = max_seq_len
        else:
            temp_new = temp + [padding_idx] * (max_seq_len - len(temp))
            temp_rev = temp[::-1] + [padding_idx] * (max_seq_len - len(temp))
            seqs_dict[key] = temp_new
            reverse_seqs_dict[key] = temp_rev
            seqs_lens_dict[key] = len(temp)

    return seqs_dict, reverse_seqs_dict, seqs_lens_dict


def save_list_with_pkl(filename, list_obj):
    with open(filename, 'wb') as f:
        pickle.dump(list_obj, f)


def load_list_with_pkl(filename):
    with open(filename, 'rb') as f:
        list_obj = pickle.load(f)

    return list_obj


def save_dict_to_pkl(pkl_filename, dict_pbj):
    with open(pkl_filename, 'wb') as f:
        pickle.dump(dict_pbj, f)


def load_dict_from_pkl(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        dict_obj = pickle.load(f)

    return dict_obj


def get_num_sessions(sessions_dict):
    num_sessions = 0
    for value in sessions_dict.values():
        num_sessions += len(value)

    return num_sessions


def process_users_seqs(users_seqs_dict, padding_idx, max_seq_len):
    processed_seqs_dict = {}
    reverse_seqs_dict = {}
    for key, seq in users_seqs_dict.items():
        if len(seq) >= max_seq_len:
            temp_seq = seq[-max_seq_len:]
            temp_rev_seq = temp_seq[::-1]
        else:
            temp_seq = seq + [padding_idx] * (max_seq_len - len(seq))
            temp_rev_seq = seq[::-1] + [padding_idx] * (max_seq_len - len(seq))
        processed_seqs_dict[key] = temp_seq
        reverse_seqs_dict[key] = temp_rev_seq

    return processed_seqs_dict, reverse_seqs_dict


def reverse_users_seqs(processed_users_seqs_dict, padding_idx, max_seq_len):
    reversed_users_seqs_dict = {}
    for key, seq in processed_users_seqs_dict.items():
        for idx in range(len(seq)):
            if seq[idx] == padding_idx:
                actual_seq = seq[:idx]
                reversed_users_seqs_dict[key] = actual_seq[::-1] + [padding_idx] * (max_seq_len - idx)
                break

    return reversed_users_seqs_dict


def gen_users_seqs_masks(users_seqs_dict, padding_idx):
    users_seqs_masks_dict = {}
    for key, seq in users_seqs_dict.items():
        temp_seq = []
        for poi in seq:
            if poi != padding_idx:
                temp_seq.append(1)
            else:
                temp_seq.append(0)
        users_seqs_masks_dict[key] = temp_seq

    return users_seqs_masks_dict


def haversine_distance(lon1, lat1, lon2, lat2):
    """Haversine distance"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371

    return c * r


def euclidean_distance(lon1, lat1, lon2, lat2):
    """Euclidean distance"""

    return np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)


def gen_geo_seqs_adjs_dict(users_seqs_dict, pois_coos_dict, max_seq_len, padding_idx, eta=1, distance_threshold=2.5, distance_type="haversine"):
    """Generate geographical sequential adjacency dictionary"""
    geo_adjs_dict = {}
    for key, seq in users_seqs_dict.items():
        geo_adj = np.zeros(shape=(max_seq_len, max_seq_len))
        actual_seq = []
        for item in seq:
            if item != padding_idx:
                actual_seq.append(item)
        actual_seq_len = len(actual_seq)
        for i in range(actual_seq_len):
            for j in range(i + 1, actual_seq_len):
                l1 = actual_seq[i]
                l2 = actual_seq[j]
                lat1, lon1 = pois_coos_dict[l1]
                lat2, lon2 = pois_coos_dict[l2]
                if distance_type == "haversine":
                    dist = haversine_distance(lon1, lat1, lon2, lat2)
                elif distance_type == "euclidean":
                    dist = euclidean_distance(lon1, lat1, lon2, lat2)
                if 0 < dist <= distance_threshold:
                    geo_influence = np.exp(-eta * (dist ** 2))
                    geo_adj[i, j] = geo_influence
                    geo_adj[j, i] = geo_influence
        geo_adjs_dict[key] = geo_adj

    return geo_adjs_dict


def create_user_poi_adj(users_seqs_dict, num_users, num_pois, padding_idx):
    """Create user-POI interaction matrix"""
    R = sp.dok_matrix((num_users, num_pois), dtype=np.float)
    for userID, seq in users_seqs_dict.items():
        for itemID in seq:
            if itemID != padding_idx:
                itemID = itemID - num_users
                R[userID, itemID] = 1
            else:
                break

    return R, R.T


def gen_sparse_A(users_seqs_dict, num_users, num_pois, padding_idx):
    """Generate sparse user-POI adjacent matrix"""
    R, R_T = create_user_poi_adj(users_seqs_dict, num_users, num_pois, padding_idx)
    A = sp.dok_matrix((num_users + num_pois, num_users + num_pois), dtype=float)
    A[:num_users, num_users:] = R
    A[num_users:, :num_users] = R_T
    A_sparse = A.tocsr()

    return A_sparse


def normalized_adj(adj):
    """Normalize adjacent matrix for GCN"""
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1/2).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj = d_mat_inv * adj * d_mat_inv

    return norm_adj


def gen_local_graph(adj):
    """Add self loop"""
    G = normalized_adj(adj + sp.eye(adj.shape[0]))

    return G


def gen_sparse_H(sessions_dict, num_pois, num_sessions, start_poiID):
    """Generate sparse incidence matrix for hypergraph"""
    H = np.zeros(shape=(num_pois, num_sessions))
    sess_idx = 0
    for key, sessions in sessions_dict.items():
        for session in sessions:
            for poiID in session:
                new_poiID = poiID - start_poiID
                H[new_poiID, sess_idx] = 1
            sess_idx += 1
    assert sess_idx == num_sessions
    H = sp.csr_matrix(H)

    return H


def gen_HG_from_sparse_H(H, conv="sym"):
    """Generate hypergraph with sparse incidence matrix"""
    n_edge = H.shape[1]
    W = sp.eye(n_edge)

    HW = H.dot(W)
    DV = sp.csr_matrix(HW.sum(axis=1)).astype(float)
    DE = sp.csr_matrix(H.sum(axis=0)).astype(float)
    invDE1 = DE.power(-1)
    invDE1_ = sp.diags(invDE1.toarray()[0])
    HT = H.T

    if conv == "sym":
        invDV2 = DV.power(n=-1 / 2)
        invDV2_ = sp.diags(invDV2.toarray()[:, 0])
        HG = invDV2_ * H * W * invDE1_ * HT * invDV2_
    elif conv == "asym":
        invDV1 = DV.power(-1)
        invDV1_ = sp.diags(invDV1.toarray()[:, 0])
        HG = invDV1_ * H * W * invDE1_ * HT

    return HG


def transform_csr_matrix_to_tensor(csr_matrix):
    """Transform csr matrix to tensor"""
    coo = csr_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    sp_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return sp_tensor

