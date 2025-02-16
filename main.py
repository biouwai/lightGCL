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

device = 'cuda:' + args.cuda

logging.basicConfig(filename='output.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def train_model(n, d, l, temp, batch_user, epoch_no, lambda_1, lambda_2, dropout, lr, svd_q):
    # 第n个参数组合
    print('组合',n,'开始训练---')
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

    # print('Data loaded.')

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
    adj_norm = adj_norm.coalesce().cuda(torch.device(device))
    # print('Adj matrix normalized.')

    # perform svd reconstruction
    adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda(torch.device(device))
    # print('Performing SVD...')
    svd_u,s,svd_v = torch.svd_lowrank(adj, q=svd_q)
    u_mul_s = svd_u @ (torch.diag(s))
    v_mul_s = svd_v @ (torch.diag(s))
    del s
    # print('SVD done.')

    # process test set
    max_user_id = max(train.shape[0], test.shape[0])
    test_labels = [[] for i in range(max_user_id)]
    for i in range(len(test.data)):
        row = test.row[i]
        col = test.col[i]
        test_labels[row].append(col)
    # print('Test data processed.')

    loss_list = []
    loss_r_list = []
    loss_s_list = []
    max_auc = 0
    max_aucr = 0
    max_epoch = 0

    model = LightGCL(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device)
    model.cuda(torch.device(device))
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
            uids = uids.long().cuda(torch.device(device))
            pos = pos.long().cuda(torch.device(device))
            neg = neg.long().cuda(torch.device(device))
            iids = torch.concat([pos, neg], dim=0)

            # feed
            optimizer.zero_grad()
            loss, loss_r, loss_s= model(uids, iids, pos, neg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().item()
            epoch_loss_r += loss_r.cpu().item()
            epoch_loss_s += loss_s.cpu().item()
            torch.cuda.empty_cache()

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
            test_uids_input = torch.LongTensor(test_uids).cuda(torch.device(device))
            predictions = model(test_uids_input,None,None,None,test=True)
            auc,aupr = metrics(test_uids,predictions,test_labels)
            if (auc + aupr) > (max_auc+max_aucr):
                max_auc = auc
                max_aucr = aupr
                max_epoch = epoch
                # print('epoch:',epoch,'AUC:',auc,'AUPR:',aupr)

    return max_epoch, max_auc, max_aucr

            # print('------------------------------------')
            # print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_r:',epoch_loss_r,'Loss_s:',epoch_loss_s)

# 定义超参数网格
param_grid = {
    'd': [32, 64, 128, 256],  # 嵌入维度
    'l': [2, 3, 4, 5],  # GNN 层数
    'temp': [1, 3, 5, 7, 9],  # 温度参数
    'batch_user': [128, 256, 512],  # 用户批次大小
    'epoch_no': [700],  # 训练轮数
    'lambda_1': [0.0001, 0.001, 0.01, 0.1],  # 正则化参数
    'lambda_2': [1e-5, 1e-4, 1e-3, 1e-2],  # 正则化参数
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.4],  # Dropout 概率
    'lr': [0.0001],  # 学习率
    'svd_q': [3, 5, 7, 9]  # SVD 分解的秩
}


best_auc = 0
best_aupr = 0
best_params = None
n = 0


for params in ParameterGrid(param_grid):
    #超参数
    # d = 64  # 嵌入维度
    # l = 3  # GNN 层数
    # temp = 5  # 温度参数
    # batch_user = 256  # 用户批次大小
    # epoch_no = 200  # 训练轮数
    # max_samp = 50  # 最大采样数
    # lambda_1 = 0.001  # 正则化参数
    # lambda_2 = 1e-4  # 正则化参数
    # dropout = 0.1  # Dropout 概率
    # lr = 0.001  # 学习率
    # decay = 1e-5  # 权重衰减系数
    # svd_q = 5  # SVD 分解的秩
    d = params['d']
    l = params['l']
    temp = params['temp']
    batch_user = params['batch_user']
    epoch_no = params['epoch_no']
    lambda_1 = params['lambda_1']
    lambda_2 = params['lambda_2']
    dropout = params['dropout']
    lr = params['lr']
    svd_q = params['svd_q']

    n = n + 1
    from datetime import datetime

    # 获取当前时间
    now = datetime.now()

    print('date',now,'n',n,'params',params)
    epoch_no, auc, aupr = train_model(n, d, l, temp, batch_user, epoch_no, lambda_1, lambda_2, dropout, lr, svd_q)

    # 同时考虑 AUC 和 AUPR 来更新最佳参数
    if (auc + aupr) > (best_auc + best_aupr):
        best_auc = auc
        best_aupr = aupr
        best_params = params
        print('epoch:',epoch_no,'AUC:',best_auc,'AUPR:',best_aupr,'params:',best_params)
        logging.info('n: %d epoch: %s AUC: %s AUPR: %s params: %s', n, epoch_no, best_auc, best_aupr, best_params)

print("Best epoch:", epoch_no)
print("Best AUC:", best_auc)
print("Best AUPR:", best_aupr)
print("Best parameters:", best_params)