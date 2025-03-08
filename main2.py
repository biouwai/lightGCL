
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
    lambda_1=0.01
    lr=0.001
    svd_q=216
    temp=0.5
    # 指标
    max_auc = 0
    max_aucr = 0
    max_epoch = 0

    df1 = pd.read_csv("newSet/drugSimMat.csv",header=None)
    df2 = pd.read_csv("newSet/LncDrug_edge.csv")
    df3 = pd.read_csv("newSet/MiDrug_edge.csv")
    df = pd.concat([df3, df2], ignore_index=True)

    ncRNA_list = sorted(df['ncRNA_Name'].unique())
    drug_list = sorted(df['Drug_Name'].unique())

    ncRNA_id_map = {name:i for i,name in enumerate(ncRNA_list)}
    drug_id_map = {name:i for i,name in enumerate(drug_list)}
    # 生成交互三元组
    rows = df['ncRNA_Name'].map(ncRNA_id_map).values
    cols = df['Drug_Name'].map(drug_id_map).values
    data1 = np.ones(len(df))  # 所有关联标记为1
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

    # 创建稀疏矩阵函数
    def create_sparse_matrix(indices, rows, cols, data1):
        return sp.coo_matrix(
            (data1[indices], (rows[indices], cols[indices])),
            shape=(len(ncRNA_list), len(drug_list)),
            dtype=np.float32
        )

    # 生成训练/测试矩阵
    train = create_sparse_matrix(train_idx, rows, cols, data1)
    test = create_sparse_matrix(test_idx, rows, cols, data1)

    train1 = train.copy()
    print('train:',train.sum())

    # 转换为CSR格式
    train_csr = train.tocsr()
    test_csr = test.tocsr()

    # 保存为txt文件
    def save_sparse(matrix, filename):
        with open(filename, 'w') as f:
            for row, col in zip(matrix.row, matrix.col):
                f.write(f"{row} {col} 1\n")

    save_sparse(train, 'dataset/rrtrain_x.txt')
    save_sparse(test, 'dataset/rrtest_x.txt')

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
    # svd_u,s,svd_v = torch.svd_lowrank(adj, q=svd_q)
    df1 = pd.read_csv("newSet/drugSimMat.csv", header=None)
    drug_sim_mat = torch.tensor(df1.values, dtype=torch.float32)

    adj_dense_drug_sim = torch.matmul(adj_dense, drug_sim_mat)
    method='full_svd'
    svd_u,s,svd_v = matrix_decomposition(adj_dense_drug_sim, method='full_svd', q=svd_q)
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
    



