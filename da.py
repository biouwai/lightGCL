# import numpy as np
# import scipy.sparse as sp


# def read_file_to_sparse_matrix(file_path):
#     rows = []
#     cols = []
#     data = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             parts = line.strip().split()
#             if len(parts) == 3:
#                 row, col, value = map(int, parts)
#                 rows.append(row)
#                 cols.append(col)
#                 data.append(value)
#     n_rows = max(rows) + 1
#     n_cols = max(cols) + 1
#     # 创建 COO 稀疏矩阵
#     coo_matrix = sp.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols), dtype=np.float32)
#     # 转换为 CSR 稀疏矩阵
#     csr_matrix = coo_matrix.tocsr()
#     return coo_matrix, csr_matrix


# # 假设文件名为 data.txt
# file_path = 'data/rtrain_0.txt'
# train, train_csr = read_file_to_sparse_matrix(file_path)


# print("COO Sparse Matrix (train):")
# print(train)


# print("\nCSR Sparse Matrix (train_csr):")
# print(train_csr)

# def AUC(label, pre):
#     # 计算正样本和负样本的索引，以便索引出之后的概率值
#     pos = [i for i in range(len(label)) if label[i] == 1]   #正样本索引
#     neg = [i for i in range(len(label)) if label[i] == 0]   #负样本索引
#     auc = 0
#     for i in pos:
#         for j in neg:
#             if pre[i] > pre[j]:
#                 auc += 1
#             elif pre[i] == pre[j]:
#                 auc += 0.5

#     return auc / (len(pos) * len(neg))

# if __name__ == '__main__':
#     label = [1, 0, 0, 0, 1, 0, 1, 0]
#     pre = [0.9, 0.8, 0.3, 0.1, 0.4, 0.9, 0.66, 0.7]
#     print(AUC(label, pre))

# print(5/8)

# import numpy as np
# from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# uids = np.array([0, 1, 2, 3, 4])
# labels = [[], [0,1,3], [1,2,3,4], [1,2], [0,1,2]]
# train_labels = [[], [0,1], [1,4], [], [0,1,2]]
# test_labels = [[], [3], [2], [1,2],[]]


# predictions = np.array([[0.2, 0.8, 0.2, 0.4],
#                      [0.3, 0.0, 0.9, 0.0],
#                      [0.1, 0.0, 0.0, 0.3],
#                      [0.6, 0.3, 0.2, 0.9],
#                      [0.1, 0.2, 0.3, 0.9]])

# num_items = predictions.shape[1]
# flat_predictions = predictions.flatten()
# flat_labels = np.zeros(len(flat_predictions))

# # 明确将 test_labels 中的元素与 uids 对应
# for user_id, labels in enumerate(test_labels):
#     for item_id in labels:
#         # 计算在 flat_labels 中的正确索引
#         index = user_id * num_items + item_id
#         if index < len(flat_labels):
#             flat_labels[index] = 1

# # 计算 AUC
# auc_score = roc_auc_score(flat_labels, flat_predictions)
# # 计算AUPR
# precision, recall, _ = precision_recall_curve(flat_labels, flat_predictions)
# aupr_score = auc(recall, precision)

# print("AUC:", auc_score)
# print("AUPR:", aupr_score)


# print("flat_labels:", flat_labels)
# print("flat_predictions:", flat_predictions)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, emb_size):
        super(MatrixFactorization, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        pred = (user_emb * item_emb).sum(dim=1)
        return pred


# def recall_and_precision(predictions, labels):
#     recall_list = []
#     precision_list = []
#     for i in range(len(predictions)):
#         pred_sorted = np.argsort(predictions[i])[::-1]
#         true_positive = set(labels[i])
#         top_k = pred_sorted[:len(true_positive)]
#         true_positive_and_retrieved = set(top_k) & true_positive
#         recall = len(true_positive_and_retrieved) / len(true_positive) if len(true_positive) > 0 else 0
#         precision = len(true_positive_and_retrieved) / len(top_k) if len(top_k) > 0 else 0
#         recall_list.append(recall)
#         precision_list.append(precision)
#     return np.mean(recall_list), np.mean(precision_list)


def train(model, optimizer, criterion, uids, train_labels, epochs=100, lr=0.01):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user_id in range(len(uids)):
            user = torch.LongTensor([user_id])
            items = torch.LongTensor(train_labels[user_id])
            target = torch.ones(len(items))
            optimizer.zero_grad()
            predictions = model(user.repeat(len(items)), items)
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss}")


def evaluate(model, uids, test_labels):
    model.eval()
    predictions = []
    with torch.no_grad():
        for user_id in range(len(uids)):
            user = torch.LongTensor([user_id])
            all_items = torch.arange(len(test_labels[user_id]))
            scores = model(user.repeat(len(all_items)), all_items)
            predictions.append(scores.numpy())
    return predictions


def calculate_aupr(predictions, labels):
    aupr_list = []
    for i in range(len(predictions)):
        pred_sorted = np.argsort(predictions[i])[::-1]
        true_positive = set(labels[i])
        precision = []
        recall = []
        correct = 0
        for k in range(1, len(pred_sorted) + 1):
            top_k = set(pred_sorted[:k])
            true_positive_and_retrieved = top_k & true_positive
            correct += len(true_positive_and_retrieved)
            if len(true_positive) > 0:
                recall.append(correct / len(true_positive))
                precision.append(correct / k)
        if len(precision) > 0 and len(recall) > 0:
            aupr = np.trapz(precision, recall)
            aupr_list.append(aupr)
    return np.mean(aupr_list)


# 数据处理
uids = np.array([0, 1, 2, 3, 4])
labels = [[], [0, 1, 3], [1, 2, 3, 4], [1, 2], [0, 1, 2]]
train_labels = [[], [0, 1], [1, 4], [], [0, 1, 2]]
test_labels = [[], [3], [2], [1, 2], []]
num_users = len(uids)
num_items = max(max(item for sublist in labels for item in sublist) + 1 if labels else 0,
              max(item for sublist in train_labels for item in sublist) + 1 if train_labels else 0,
              max(item for sublist in test_labels for item in sublist) + 1 if test_labels else 0)


# 模型初始化和训练
model = MatrixFactorization(num_users, num_items, emb_size=10)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
train(model, optimizer, criterion, uids, train_labels, epochs=200, lr=0.01)


# 评估
predictions = evaluate(model, uids, test_labels)
aupr = calculate_aupr(predictions, test_labels)
print(f"AUPR: {aupr}")
