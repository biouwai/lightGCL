
from datetime import date
import logging
import numpy as np
from sklearn.model_selection import ParameterGrid
import torch
import scipy.sparse as sp
from model import LightGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
from parser import args
import torch.utils.data as data
from utils import TrnData
import warnings

warnings.filterwarnings("ignore")

device = 'cpu'

logging.basicConfig(filename='output.log', level=logging.INFO, format='%(asctime)s - %(message)s')




# ----


# 0/5
# batch_user=128
# d=128
# dropout=0.0
# epoch_no=120
# l=4
# lambda_1=0.01
# lambda_2=0.0001
# lr=0.001
# svd_q=7
# temp=3

# 上双9 0/5
# batch_user=128
# d=128
# dropout=0.0
# epoch_no=120
# l=4
# lambda_1=0.01
# lambda_2=0.0001
# lr=0.001
# svd_q=7
# temp=3

# 双9 0/5
# batch_user=128
# d=256
# dropout=0.1
# epoch_no=120
# l=4
# lambda_1=0.001
# lambda_2=0.0001
# lr=0.001
# svd_q=3
# temp=3

# 0/5
# batch_user=128
# d=128
# dropout=0.0
# epoch_no=120
# l=4
# lambda_1=0.0001
# lambda_2=1e-05
# lr=0.001
# svd_q=9
# temp=9

# 实验参数
# 2/5
batch_user=256
d=256
dropout=0.0
epoch_no=140
l=2
lambda_1=0.0
lambda_2=0.0001
lr=0.001
svd_q=3
temp=9

# 数据加载
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

train_path = 'dataset/rtrain_0.txt'
train, train_csr = read_file_to_sparse_matrix(train_path)
test_path = 'dataset/rtest_0.txt'
test, test_csr = read_file_to_sparse_matrix(test_path)

print('Data loaded.')

# print('user_num:',train.shape[0],'item_num:',train.shape[1],'lambda_1:',lambda_1,'lambda_2:',lambda_2,'temp:',temp,'q:',svd_q)

# epoch_user = min(train.shape[0], 30000)

# normalizing the adj matrix
rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)

# construct data loader
train = train.tocoo()
train_data = TrnData(train)
train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce()
print('Adj matrix normalized.')

# perform svd reconstruction
adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce()
# print('Performing SVD...')
svd_u,s,svd_v = torch.svd_lowrank(adj, q=svd_q)
u_mul_s = svd_u @ (torch.diag(s))
v_mul_s = svd_v @ (torch.diag(s))
del s
print('SVD done.')

# process test set
max_user_id = max(train.shape[0], test.shape[0])
test_labels = [[] for i in range(max_user_id)]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)
print('Test data processed.')

loss_list = []
loss_r_list = []
loss_s_list = []
max_auc = 0
max_aucr = 0
max_epoch = 0

model = LightGCL(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device)

optimizer = torch.optim.Adam(model.parameters(),weight_decay=0,lr=lr)
current_lr = lr

for epoch in range(epoch_no):
    # if (epoch+1)%50 == 0:
    #     torch.save(model.state_dict(),'saved_model/saved_model_epoch_'+str(epoch)+'.pt')
    #     torch.save(optimizer.state_dict(),'saved_model/saved_optim_epoch_'+str(epoch)+'.pt')

    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    train_loader.dataset.neg_sampling()
    for i, batch in enumerate(train_loader):
        uids, pos, neg = batch
        uids = uids.long()
        pos = pos.long()
        neg = neg.long()
        iids = torch.concat([pos, neg], dim=0)

        # feed
        optimizer.zero_grad()
        loss, loss_r, loss_s= model(uids, iids, pos, neg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()

    batch_no = len(train_loader)
    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    epoch_loss_s = epoch_loss_s/batch_no
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)

    if epoch % 1 == 0:  # test every 5 epochs
        test_uids = np.array([i for i in range(adj_norm.shape[0])])
        batch_no = int(np.ceil(len(test_uids)/batch_user))
        test_uids_input = torch.LongTensor(test_uids)
        predictions = model(test_uids_input,None,None,None,test=True)
        auc,aupr = metrics(test_uids,predictions,test_labels)
        # print('------------------------------------')
        if (auc + aupr) > (max_auc+max_aucr):
            max_auc = auc
            max_aucr = aupr
            max_epoch = epoch
        print('epoch:',epoch,'AUC:',auc,'AUPR:',aupr)
print('MAX,epoch:',max_epoch,'AUC:',max_auc,'AUPR:',max_aucr)

 