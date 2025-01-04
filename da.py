import numpy as np
import scipy.sparse as sp


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


# 假设文件名为 data.txt
file_path = 'data/rtrain_0.txt'
train, train_csr = read_file_to_sparse_matrix(file_path)


print("COO Sparse Matrix (train):")
print(train)


print("\nCSR Sparse Matrix (train_csr):")
print(train_csr)