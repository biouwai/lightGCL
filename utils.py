import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def metrics(uids, predictions, test_labels):
    # 步骤 1: 将 numpy.int32 转换为 int
    test_labels = [[int(item) for item in sublist] for sublist in test_labels]
    num_items = predictions.shape[1]
    # 修改：使用 detach().numpy() 从计算图中分离并转换为 numpy 数组
    flat_predictions = predictions.flatten().detach().numpy()
    flat_labels = np.zeros(len(flat_predictions))

    # 明确将 test_labels 中的元素与 uids 对应
    for user_id, labels in enumerate(test_labels):
        for item_id in labels:
            # 计算在 flat_labels 中的正确索引
            index = user_id * num_items + item_id
            if index < len(flat_labels):
                flat_labels[index] = 1

    # 计算正样本的数量
    num_positive = int(np.sum(flat_labels))

    # 找出正样本的索引
    positive_indices = np.where(flat_labels == 1)[0]
    # 找出负样本的索引
    negative_indices = np.where(flat_labels == 0)[0]

    # 随机选取和正样本数量相同的负样本索引
    np.random.shuffle(negative_indices)
    selected_negative_indices = negative_indices[:num_positive]

    # 合并正样本和选取的负样本索引
    selected_indices = np.concatenate((positive_indices, selected_negative_indices))
    selected_indices = np.sort(selected_indices)

    # 截取相应的预测值和标签
    selected_flat_predictions = flat_predictions[selected_indices]
    selected_flat_labels = flat_labels[selected_indices]

    # 计算 AUC
    auc_score = roc_auc_score(selected_flat_labels, selected_flat_predictions)
    # 计算 AUPR
    precision, recall, _ = precision_recall_curve(selected_flat_labels, selected_flat_predictions)
    aupr_score = auc(recall, precision)

    return auc_score, aupr_score

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs =  emb[cols] * torch.unsqueeze(sp.values(),dim=1)
    result = torch.zeros((sp.shape[0],emb.shape[1])).cuda(torch.device(device))
    result.index_add_(0, rows, col_segs)
    return result

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]