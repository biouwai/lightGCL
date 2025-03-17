import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def metrics(uids, predictions, test_labels):
    # 步骤 1: 将 numpy.int32 转换为 int
    test_labels = [[int(item) for item in sublist] for sublist in test_labels]
    # 获取物品数量（预测矩阵的列数）
    num_items = predictions.shape[1]
    # 将预测值从 GPU 张量复制到 CPU，并转换为 NumPy 数组
    flat_predictions = predictions.flatten().detach().cpu().numpy()
    # 初始化标签数组，所有值默认为 0（负样本）
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
    selected_negative_indices = negative_indices[:(num_positive)]

    # 合并正样本和选取的负样本索引
    selected_indices = np.concatenate((positive_indices, selected_negative_indices))
    selected_indices = np.sort(selected_indices)

    # 截取相应的预测值和标签
    selected_flat_predictions = flat_predictions[selected_indices]
    selected_flat_labels = flat_labels[selected_indices]

    # 避免出错
    selected_flat_predictions = np.nan_to_num(selected_flat_predictions, nan=0)
    selected_flat_labels = np.nan_to_num(selected_flat_labels, nan=0)

    # 计算 AUC
    auc_score = roc_auc_score(selected_flat_labels, selected_flat_predictions)
    # 计算 AUPR
    precision, recall, _ = precision_recall_curve(selected_flat_labels, selected_flat_predictions)
    aupr_score = auc(recall, precision)

    return auc_score, aupr_score

# 将 SciPy 稀疏矩阵转换为 PyTorch 稀疏张量
def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# 如果 dropout 概率为 0，则直接返回原矩阵。否则，随机丢弃稀疏矩阵中的非零值。
def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    # 为每个正样本生成一个负样本。
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

    # 根据索引 idx，返回对应的用户 ID、正样本物品 ID 和负样本物品 ID
    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

# 解析文件内容，提取用户 ID、物品 ID 和交互值。构造 COO 格式的稀疏矩阵
def load_sparse(filename):

    rows = []
    cols = []
    data = []
    n_rows = 0
    n_cols = 0
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("# Shape:"):
                # 解析矩阵形状
                parts = line.strip().split()
                n_rows = int(parts[2])
                n_cols = int(parts[3])
            else:
                # 解析非零元素
                parts = line.strip().split()
                if len(parts) == 3:
                    row, col, value = map(int, parts)
                    rows.append(row)
                    cols.append(col)
                    data.append(value)
    
    # 创建稀疏矩阵
    sparse_matrix = coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols), dtype=np.float32)
    return sparse_matrix

# 解析文件内容，提取用户 ID、物品 ID 和交互值。构造 COO 格式的稀疏矩阵，并转换为 CSR 格式以提高计算效率。
def read_file_to_sparse_matrix(file_path):
    rows = []
    cols = []
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                row, col, value = map(int, parts)
                rows.append(row)
                cols.append(col)
                data.append(value)
    n_rows = max(rows) + 1
    n_cols = max(cols) + 1
    # 创建 COO 稀疏矩阵
    coo_matrix = sp.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols), dtype=np.float32)
    # 转换为 CSR 稀疏矩阵
    csr_matrix = coo_matrix.tocsr()
    return coo_matrix, csr_matrix

# 对输入矩阵进行分解，支持多种分解方法（如 SVD、PCA、QR、NMF 等）
def matrix_decomposition(adj, method='full_svd', q=None):
    if method == 'full_svd':
        U, S, Vh = torch.linalg.svd(adj)
        return U[:, :q], S[:q], Vh[:q, :].T  # Return V instead of V^T
    elif method == 'lowrank':
        U, S, V = torch.svd_lowrank(adj, q=q)
        return U, S, V  # No need to transpose
    elif method == 'pca':
        # PCA decomposition
        U, S, V = torch.pca_lowrank(adj.to_dense(), q=q)
        return U, S, V  # Transpose V to match expected shape (n x q)
    elif method == 'qr':
        Q, R = torch.linalg.qr(adj.to_dense())
        # Extract the first q columns of Q and R
        U_qr = Q[:, :q]  # Shape: (m, q)
        S_qr = torch.diag(R[:q, :q])  # Singular values (diagonal of R)
        V_qr = R[:q, :].T  # Transpose R to match expected shape (n, q)
        return U_qr, S_qr, V_qr
    elif method == 'nmf':
        from sklearn.decomposition import NMF
        # Convert sparse tensor to dense tensor first
        adj_dense = adj.to_dense().numpy()
        model = NMF(n_components=q)
        W = model.fit_transform(adj_dense)  # Fit NMF on dense matrix
        H = model.components_
        return torch.Tensor(W), None, torch.Tensor(H.T)  # Return H^T as V
    else:
        raise ValueError(f"Unsupported decomposition method: {method}")
    
# class TrnData(data.Dataset):
#     def __init__(self, coomat, drug_sim=None):
#         self.rows = coomat.row
#         self.cols = coomat.col
#         self.dokmat = coomat.todok()
#         self.drug_sim = drug_sim  # 添加药物相似度矩阵
#         self.negs = np.zeros((len(self.rows), 5), dtype=np.int32)  # 每个正样本5个负样本

#     def neg_sampling(self):
#         for i in range(len(self.rows)):
#             u = self.rows[i]
#             pos_i = self.cols[i]
#             valid_negs = []
            
#             # 困难负采样（基于相似度的前2个）
#             if self.drug_sim is not None:
#                 sim_scores = self.drug_sim[pos_i]
#                 sorted_items = np.argsort(-sim_scores)  # 降序排序
#                 for item in sorted_items:
#                     if item != pos_i and (u, item) not in self.dokmat:
#                         valid_negs.append(item)
#                         if len(valid_negs) >= 2:
#                             break
            
#             # 随机补充剩余3个
#             while len(valid_negs) < 5:
#                 i_neg = np.random.randint(self.dokmat.shape[1])
#                 if (u, i_neg) not in self.dokmat and i_neg not in valid_negs:
#                     valid_negs.append(i_neg)
            
#             self.negs[i] = valid_negs[:5]

#     def __len__(self):
#         return len(self.rows)
#     def __getitem__(self, idx):
#         return self.rows[idx], self.cols[idx], self.negs[idx]  # 返回5个负样本
    
# def spmm(sp, emb, device):
#     sp = sp.coalesce()
#     cols = sp.indices()[1]
#     rows = sp.indices()[0]
#     col_segs =  emb[cols] * torch.unsqueeze(sp.values(),dim=1)
#     result = torch.zeros((sp.shape[0],emb.shape[1])).cuda(torch.device(device))
#     result.index_add_(0, rows, col_segs)
#     return result