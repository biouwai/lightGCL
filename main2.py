
from datetime import date
import logging
import numpy as np
from sklearn.model_selection import ParameterGrid
import torch
import scipy.sparse as sp
from model2 import LightGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor,read_file_to_sparse_matrix,matrix_decomposition
from parser import args
import torch.utils.data as data
from utils import TrnData
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pubchempy as pcp
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix


warnings.filterwarnings("ignore")

device = 'cpu'

logging.basicConfig(filename='duibi.log', level=logging.INFO, format='%(asctime)s - %(message)s')

max_auc_list = []
max_aucr_list = []
for i in range(1):
    # 实验参数
    batch_user=256
    d=64
    dropout=0.2
    epoch_no=120
    l=2
    lambda_2=0.00001
    lambda_1=1
    lr=0.001
    svd_q=121
    temp=0.5
    # 指标
    max_auc = 0
    max_aucr = 0
    max_epoch = 0


    def load_sparse(filename):
        """
        从文件加载稀疏矩阵。
        
        参数:
            filename (str): 文件路径。
        
        返回:
            scipy.sparse.coo_matrix: 还原的稀疏矩阵。
        """
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

    # 示例：加载 train 和 test
    train = load_sparse('dataset/rrtrain_x.txt')
    train_csr = train.tocsr()
    test = load_sparse('dataset/rrtest_x.txt')

    train1 = train.copy()
    test1 = test.copy()
    
    # normalizing the adj matrix
    rowD = np.array(train.sum(1)).squeeze()
    colD = np.array(train.sum(0)).squeeze()
    for i in range(len(train.data)):
        train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)

    # construct data loader
    train = train.tocoo()
    test = test.tocoo()
    train_data = TrnData(train)
    train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)

    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
    adj_norm = adj_norm.coalesce()

    # perform svd reconstruction
    adj = scipy_sparse_mat_to_torch_sparse_tensor(train1).coalesce()
    test1 = scipy_sparse_mat_to_torch_sparse_tensor(test1).coalesce()
    # 4. 转换为密集张量
    adj_dense = adj.to_dense()
    adj_dense1 = test1.to_dense()

    svd_u,s,svd_v = matrix_decomposition(adj_dense1, method='full_svd', q=svd_q)
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

    loss_list = []
    loss_r_list = []
    loss_s_list = []


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


        if epoch % 5 == 0:  # test every 5 epochs
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
            # print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_r:',epoch_loss_r,'Loss_s:',epoch_loss_s)
            print('max_auc:',auc,'max_aucr:',aupr,'max_epoch:',epoch)

    print('max_auc:',max_auc,'max_aucr:',max_aucr,'max_epoch:',max_epoch)
    max_auc_list.append(max_auc)
    max_aucr_list.append(max_aucr)
    avg_auc = np.mean(max_auc_list)
    avg_aucr = np.mean(max_aucr_list)
    print('loss',loss_list[4],'loss_r',loss_r_list[4],loss_s_list[4])
print('mode: nosvd','max_auc_list',max_auc_list,'max_aucr_list:',max_aucr_list,'avg_auc:',avg_auc,'avg_aucr:',avg_aucr)
logging.info('mode: avg_auc: %s, avg_aucr: %s', avg_auc, avg_aucr)
    



