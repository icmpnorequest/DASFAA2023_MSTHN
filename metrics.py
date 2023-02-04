import numpy as np


def hit_k(y_pred, y_true, k):
    y_pred_indices = y_pred.topk(k=k).indices.tolist()
    if y_true in y_pred_indices:
        return 1
    else:
        return 0


def ndcg_k(y_pred, y_true, k):
    y_pred_indices = y_pred.topk(k=k).indices.tolist()
    if y_true in y_pred_indices:
        position = y_pred_indices.index(y_true) + 1
        return 1 / np.log2(1 + position)
    else:
        return 0


def batch_performance(batch_y_pred, batch_y_true, k):
    batch_size = batch_y_pred.size(0)
    batch_recall = 0
    batch_ndcg = 0
    for idx in range(batch_size):
        hit = hit_k(batch_y_pred[idx], batch_y_true[idx], k)
        batch_recall += hit
        ndcg = ndcg_k(batch_y_pred[idx], batch_y_true[idx], k)
        batch_ndcg += ndcg

    recall = batch_recall / batch_size
    ndcg = batch_ndcg / batch_size

    return recall, ndcg


