import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class STLayer(nn.Module):
    """Spatial-temporal enhanced graph neural network"""
    def __init__(self, num_users, num_pois, seq_len, emb_dim, num_heads, dropout, device):
        super(STLayer, self).__init__()

        self.num_users = num_users
        self.num_pois = num_pois
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.device = device

        self.pos_embeddings = nn.Embedding(seq_len + 1, emb_dim, padding_idx=0)
        self.FC_geo = nn.Linear(emb_dim, emb_dim, bias=True, device=device)
        self.MultiAttn = nn.MultiheadAttention(emb_dim, num_heads, dropout, batch_first=True, device=device)
        self.weight = nn.Parameter(torch.Tensor(emb_dim, emb_dim))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, G, nodes_embeds, batch_users_seqs, batch_users_seqs_masks, batch_users_geo_adjs, batch_users_indices):
        batch_size = batch_users_seqs.size(0)
        batch_users_geo_adjs = batch_users_geo_adjs.float()

        # generate sequence embeddings
        batch_seqs_embeds = nodes_embeds[batch_users_seqs]

        # generate position embeddings
        batch_seqs_pos = torch.arange(1, self.seq_len + 1, device=self.device)
        batch_seqs_pos = batch_seqs_pos.repeat(batch_size, 1)
        batch_seqs_pos = torch.multiply(batch_seqs_pos, batch_users_seqs_masks)
        batch_seqs_pos_embs = self.pos_embeddings(batch_seqs_pos)

        # generate geographical embeddings
        batch_seqs_geo_embeds = batch_users_geo_adjs.matmul(batch_seqs_embeds)
        batch_seqs_geo_embeds = torch.relu(self.FC_geo(batch_seqs_geo_embeds))

        # multi-head attention
        batch_seqs_total_embeds = batch_seqs_embeds + batch_seqs_pos_embs + batch_seqs_geo_embeds
        batch_seqs_mha, batch_seqs_mha_weight = self.MultiAttn(batch_seqs_total_embeds, batch_seqs_total_embeds, batch_seqs_total_embeds)
        batch_users_embeds = torch.mean(batch_seqs_mha, dim=1)

        nodes_embeds = nodes_embeds.clone()
        nodes_embeds[batch_users_indices] = batch_users_embeds

        # graph convolutional
        lconv_nodes_embeds = torch.sparse.mm(G, nodes_embeds[:-1])
        nodes_embeds[:-1] = lconv_nodes_embeds

        return nodes_embeds


class LocalGraph(nn.Module):
    """Local spatial-temporal enhanced graph neural network module"""
    def __init__(self, num_layers, num_users, num_pois, seq_len, emb_dim, num_heads, dropout, device):
        super(LocalGraph, self).__init__()

        self.num_layers = num_layers
        self.num_users = num_users
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.device = device
        self.STLayer = STLayer(num_users, num_pois, seq_len, emb_dim, num_heads, dropout, device)

    def forward(self, G, nodes_embeds, batch_users_seqs, batch_users_seqs_masks, batch_users_geo_adjs, batch_users_indices):
        nodes_embedding = [nodes_embeds]
        for layer in range(self.num_layers):
            nodes_embeds = self.STLayer(G, nodes_embeds, batch_users_seqs, batch_users_seqs_masks, batch_users_geo_adjs, batch_users_indices)
            # nodes_embeds = F.dropout(nodes_embeds, self.dropout)
            nodes_embedding.append(nodes_embeds)

        nodes_embeds_tensor = torch.stack(nodes_embedding)
        final_nodes_embeds = torch.mean(nodes_embeds_tensor, dim=0)

        return final_nodes_embeds


class HyGCN(nn.Module):
    """Hypergraph convolutional network"""
    def __init__(self, emb_dim, num_layers, num_users, dropout, device):
        super(HyGCN, self).__init__()

        self.num_layers = num_layers
        self.num_users = num_users
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.device = device

    def forward(self, x, HG):
        item_embedding = [x]
        for layer in range(self.num_layers):
            x = torch.sparse.mm(HG, x)
            # x = F.dropout(x, self.dropout)
            item_embedding.append(x)

        item_embedding_tensor = torch.stack(item_embedding)
        final_item_embedding = torch.mean(item_embedding_tensor, dim=0)

        return final_item_embedding


class MSTHN(nn.Module):
    """Our proposed Multi-view Spatial-Temporal Enhanced Hypergraph Network (MSTHN)"""
    def __init__(self, num_local_layer, num_global_layer, num_users, num_pois, seq_len, emb_dim, num_heads, dropout, device):
        super(MSTHN, self).__init__()

        self.num_nodes = num_users + num_pois + 1
        self.num_users = num_users
        self.num_pois = num_pois
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.padding_idx = num_users + num_pois
        self.device = device

        self.nodes_embeddings = nn.Embedding(self.num_nodes, emb_dim, padding_idx=self.padding_idx)
        self.pos_embeddings = nn.Embedding(seq_len + 1, emb_dim, padding_idx=0)
        self.w_1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_dim, 1))
        self.glu1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.glu2 = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        self.LocalGraph = LocalGraph(num_local_layer, num_users, num_pois, seq_len, emb_dim, num_heads, dropout, device)
        self.GlobalHyG = HyGCN(emb_dim, num_global_layer, num_users, dropout, device)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def user_temporal_pref_augmentation(self, node_embedding, session_len, reversed_sess_item, mask):
        """user temporal preference augmentation"""
        batch_size = session_len.size(0)
        seq_h = node_embedding[reversed_sess_item]
        hs = torch.div(torch.sum(seq_h, 1), session_len.unsqueeze(-1))

        batch_seqs_pos = torch.arange(1, self.seq_len + 1, device=self.device)
        batch_seqs_pos = batch_seqs_pos.repeat(batch_size, 1)
        batch_seqs_pos = torch.multiply(batch_seqs_pos, mask)
        pos_emb = self.pos_embeddings(batch_seqs_pos)

        hs = hs.unsqueeze(1).repeat(1, self.seq_len, 1)
        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        select = torch.sum(beta * seq_h, 1)

        return select

    def forward(self, G, HG, batch_users_seqs, batch_users_seqs_masks, batch_users_geo_adjs, batch_users_indices, batch_seqs_lens, batch_users_rev_seqs):
        nodes_embeds = self.nodes_embeddings.weight

        local_nodes_embs = self.LocalGraph(G, nodes_embeds, batch_users_seqs, batch_users_seqs_masks, batch_users_geo_adjs, batch_users_indices)
        local_batch_users_embs = local_nodes_embs[batch_users_indices]
        local_pois_embs = local_nodes_embs[self.num_users: -1, :]

        global_pois_embs = self.GlobalHyG(nodes_embeds[self.num_users: -1, :], HG)

        pois_embs = local_pois_embs + global_pois_embs
        fusion_nodes_embs = torch.cat([local_nodes_embs[:self.num_users], pois_embs], dim=0)
        fusion_nodes_embs = torch.cat([fusion_nodes_embs, torch.zeros(size=(1, self.emb_dim), device=self.device)], dim=0)
        batch_users_embs = self.user_temporal_pref_augmentation(fusion_nodes_embs, batch_seqs_lens, batch_users_rev_seqs, batch_users_seqs_masks)
        batch_users_embs = batch_users_embs + local_batch_users_embs

        return batch_users_embs, pois_embs


