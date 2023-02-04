# coding=utf-8
"""
@author: Yantong Lai
@description: Code of Multi-View Spatial-Temporal Enhanced Hypergraph Network for Next POI Recommendation
"""

import argparse
import time
from torch.utils.data import DataLoader
import torch.optim as optim
import random

from utils import *
from dataset import POIDataset
from metrics import batch_performance
from model import *

seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="NYC", help='NYC/TKY/Gowalla')
parser.add_argument('--num_heads', type=int, default=8, help='number of heads for multi-attention')
parser.add_argument('--max_seq_len', default=100, type=int, help='fixed sequence length')
parser.add_argument('--eta', default=1, type=int, help='control geographical influence')
parser.add_argument('--distance_threshold', default=2.5, type=float, help='distance threshold')
parser.add_argument('--distance_type', default="haversine", type=str, help='haversine/euclidean')
parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--emb_dim', type=int, default=128, help='embedding size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--decay', type=float, default=1e-5)
parser.add_argument('--num_global_layer', type=int, default=3, help='number of hypergraph convolutional layer')
parser.add_argument('--num_local_layer', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
parser.add_argument('--conv', type=str, default="sym", help='Symmetric hypergraph or asymmetric hypergraph')
parser.add_argument('--deviceID', type=int, default=0)
args = parser.parse_args()


device = torch.device("cuda:{}".format(args.deviceID) if torch.cuda.is_available() else "cpu")


def main():
    print("1. Read Arguments")
    if args.dataset == "TKY":
        NUM_USERS = 2173
        NUM_POIS = 7038
        PADDING_IDX = 9211
    elif args.dataset == "NYC":
        NUM_USERS = 834
        NUM_POIS = 3835
        PADDING_IDX = 4669
    elif args.dataset == "Gowalla":
        NUM_USERS = 5802
        NUM_POIS = 40868
        PADDING_IDX = 46670

    print("2. Load Dataset")
    pois_coos_dict = load_dict_from_pkl("datasets/{}/{}_pois_coos.pkl".format(args.dataset, args.dataset))
    train_dataset = POIDataset(data_filename="datasets/{}/train.txt".format(args.dataset),
                                      pois_coos_dict=pois_coos_dict, num_users=NUM_USERS, num_pois=NUM_POIS,
                                      max_seq_len=args.max_seq_len, padding_idx=PADDING_IDX, eta=args.eta,
                                      distance_threshold=args.distance_threshold, distance_type=args.distance_type,
                                      conv=args.conv, device=device)
    test_dataset = POIDataset(data_filename="datasets/{}/test.txt".format(args.dataset),
                                     pois_coos_dict=pois_coos_dict, num_users=NUM_USERS, num_pois=NUM_POIS,
                                     max_seq_len=args.max_seq_len, padding_idx=PADDING_IDX, eta=args.eta,
                                     distance_threshold=args.distance_threshold, distance_type=args.distance_type,
                                     conv=args.conv, device=device)

    print("3. Construct DataLoader")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    print("4. Load Model")
    model = MSTHN(args.num_local_layer, args.num_global_layer, NUM_USERS, NUM_POIS, args.max_seq_len, args.emb_dim,
                  args.num_heads, args.dropout, device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss().to(device)

    print("5. Start Training")
    Ks_list = [5, 10]
    final_results = {"Rec5": 0.0, "Rec10": 0.0, "NDCG5": 0.0, "NDCG10": 0.0}

    for epoch in range(args.num_epochs):
        print("Epoch {}/{}".format(epoch + 1, args.num_epochs))
        start_time = time.time()
        model.train()

        train_loss = 0.0
        train_recall_array = np.zeros(shape=(len(train_dataloader), len(Ks_list)))
        train_ndcg_array = np.zeros(shape=(len(train_dataloader), len(Ks_list)))
        for idx, batch in enumerate(train_dataloader):
            print("Train. Batch {}/{}".format(idx, len(train_dataloader)))

            batch_users_indices = batch["user_idx"].to(device)
            batch_users_seqs = batch["user_seq"].to(device)
            batch_users_rev_seqs = batch["user_rev_seq"].to(device)
            batch_users_seqs_lens = batch["user_seq_len"].to(device)
            batch_users_seqs_masks = batch["user_seq_mask"].to(device)
            batch_users_geo_adjs = batch["user_geo_adj"].to(device)
            batch_users_labels = batch["label"].to(device)

            optimizer.zero_grad()

            batch_users_embs, pois_embs = model(train_dataset.G, train_dataset.HG, batch_users_seqs,
                                                batch_users_seqs_masks, batch_users_geo_adjs, batch_users_indices,
                                                batch_users_seqs_lens, batch_users_rev_seqs)
            predictions = torch.matmul(batch_users_embs, pois_embs.t())
            batch_loss = criterion(predictions, batch_users_labels)
            print("Train. batch_loss: {}".format(batch_loss.item()))

            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()

            for k in Ks_list:
                recall, ndcg = batch_performance(predictions.detach().cpu(), batch_users_labels.detach().cpu(), k)
                col_idx = Ks_list.index(k)
                train_recall_array[idx, col_idx] = recall
                train_ndcg_array[idx, col_idx] = ndcg

        print("Training finishes at this epoch. It takes {} min".format((time.time() - start_time) / 60))
        print("Training loss: {}".format(train_loss / len(train_dataloader)))
        print("Training Epoch {}/{} results:".format(epoch + 1, args.num_epochs))
        for k in Ks_list:
            col_idx = Ks_list.index(k)
            print("Recall@{}: {}".format(k, np.mean(train_recall_array[:, col_idx])))
            print("NDCG@{}: {}".format(k, np.mean(train_ndcg_array[:, col_idx])))
        print("\n")

        print("Testing")
        test_loss = 0.0
        test_recall_array = np.zeros(shape=(len(test_dataloader), len(Ks_list)))
        test_ndcg_array = np.zeros(shape=(len(test_dataloader), len(Ks_list)))

        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):
                batch_users_indices = batch["user_idx"].to(device)
                batch_users_seqs = batch["user_seq"].to(device)
                batch_users_rev_seqs = batch["user_rev_seq"].to(device)
                batch_users_seqs_lens = batch["user_seq_len"].to(device)
                batch_users_seqs_masks = batch["user_seq_mask"].to(device)
                batch_users_geo_adjs = batch["user_geo_adj"].to(device)
                batch_users_labels = batch["label"].to(device)

                batch_users_embs, pois_embs = model(test_dataset.G, test_dataset.HG, batch_users_seqs,
                                                    batch_users_seqs_masks, batch_users_geo_adjs, batch_users_indices,
                                                    batch_users_seqs_lens, batch_users_rev_seqs)
                predictions = torch.matmul(batch_users_embs, pois_embs.t())
                batch_loss = criterion(predictions, batch_users_labels)
                test_loss += batch_loss.item()

                for k in Ks_list:
                    recall, ndcg = batch_performance(predictions.detach().cpu(), batch_users_labels.detach().cpu(), k)
                    col_idx = Ks_list.index(k)
                    test_recall_array[idx, col_idx] = recall
                    test_ndcg_array[idx, col_idx] = ndcg

        print("Testing finishes")
        print("Testing loss: {}".format(test_loss / len(test_dataloader)))
        print("Testing results:")
        for k in Ks_list:
            col_idx = Ks_list.index(k)
            recall = np.mean(test_recall_array[:, col_idx])
            ndcg = np.mean(test_ndcg_array[:, col_idx])
            print("Recall@{}: {}".format(k, recall))
            print("NDCG@{}: {}".format(k, ndcg))

            # update result
            if k == 5:
                if recall > final_results["Rec5"]:
                    final_results["Rec5"] = recall
                if ndcg > final_results["NDCG5"]:
                    final_results["NDCG5"] = ndcg
            elif k == 10:
                if recall > final_results["Rec10"]:
                    final_results["Rec10"] = recall
                if ndcg > final_results["NDCG10"]:
                    final_results["NDCG10"] = ndcg
        print("\n")

    print("6. Final Results")
    print(final_results)
    print("\n")


if __name__ == '__main__':
    main()


