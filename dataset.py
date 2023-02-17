import torch
from utils import *
from torch.utils.data import Dataset, DataLoader


class POIDataset(Dataset):
    def __init__(self, data_filename, pois_coos_dict, num_users, num_pois, max_seq_len, padding_idx, eta,
                 distance_threshold, distance_type, conv, device):
        # load data
        self.data = load_list_with_pkl(data_filename)
        self.sessions_dict = self.data[0]
        self.labels_dict = self.data[1]
        self.pois_coos_dict = pois_coos_dict

        # definition
        self.num_users = num_users
        self.num_pois = num_pois
        self.num_sessions = get_num_sessions(self.sessions_dict)
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.eta = eta
        self.distance_threshold = distance_threshold
        self.distance_type = distance_type
        self.conv = conv
        self.device = device

        # generate sequence and geographical sequence for users
        self.unique_users_seqs_dict, self.unique_users_seqs_lens_dict = get_unique_seqs_for_sessions(self.sessions_dict)
        self.users_seqs_dict, self.users_reverse_seqs_dict = process_users_seqs(self.unique_users_seqs_dict, self.padding_idx, self.max_seq_len)
        self.users_seqs_masks_dict = gen_users_seqs_masks(self.users_seqs_dict, self.padding_idx)
        self.users_geo_seqs_dict = gen_geo_seqs_adjs_dict(self.users_seqs_dict, self.pois_coos_dict, self.max_seq_len, self.padding_idx, eta, distance_threshold, distance_type)

        # generate local graph
        self.A = gen_sparse_A(self.users_seqs_dict, num_users, num_pois, padding_idx)
        self.G = gen_local_graph(self.A)

        # generate global hypergraph
        self.H = gen_sparse_H(self.sessions_dict, num_pois, self.num_sessions, self.padding_idx)
        self.HG = gen_HG_from_sparse_H(self.H, conv)

        # transform csr matrix to tensor
        self.G = transform_csr_matrix_to_tensor(self.G)
        self.HG = transform_csr_matrix_to_tensor(self.HG)
        self.G = self.G.to(self.device)
        self.HG = self.HG.to(self.device)

    def __len__(self):
        return len(self.sessions_dict)

    def __getitem__(self, user_idx):
        user_idx_tensor = torch.tensor(user_idx).to(self.device)
        user_seq = torch.tensor(self.users_seqs_dict[user_idx]).to(self.device)
        user_rev_seq = torch.tensor(self.users_reverse_seqs_dict[user_idx]).to(self.device)
        user_seq_len = torch.tensor(self.unique_users_seqs_lens_dict[user_idx]).to(self.device)
        user_geo_adj = self.users_geo_seqs_dict[user_idx]
        user_geo_adj = torch.from_numpy(user_geo_adj).to(self.device)
        user_seq_mask = torch.tensor(self.users_seqs_masks_dict[user_idx]).to(self.device)
        label = torch.tensor(self.labels_dict[user_idx] - self.num_users).to(self.device)

        sample = {"user_idx": user_idx_tensor,
                  "user_seq": user_seq,
                  "user_rev_seq": user_rev_seq,
                  "user_seq_len": user_seq_len,
                  "user_geo_adj": user_geo_adj,
                  "user_seq_mask": user_seq_mask,
                  "label": label
                  }

        return sample


