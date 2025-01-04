import numpy as np
import torch
import pickle
from model import LightGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
import pandas as pd
from parser import args
from tqdm import tqdm
import time
import torch.utils.data as data
from utils import TrnData
import warnings

warnings.filterwarnings("ignore")

# device = 'cuda:' + args.cuda
device = 'cpu'

# 一、定义超参数；二、加载数据及预处理；三、奇异值分解；
# 四、处理测试集数据； 五、初始化相关列表及模型优化器
# 六、模型训练循环；七、最终测试及结果保存

# 一、定义超参数 
# hyperparameters
# 嵌入维度 -->  将物品、用户数据映射到低维，使模型能更好的处理。一般嵌入维度越高，所包涵的信息也就越丰富
d = args.d
# GNN层数
l = args.gnn_layer
# 温度参数 --> 调节概率分布形态，temp越大，概率分布更相近，可以得到多样化的结果；temp越小，得到输出会更确定
temp = args.temp
# 用户批次大小 --> 一次训练的用户数量
batch_user = args.batch
# 训练轮数 --> 一轮训练（epoch）指的是模型完整地遍历一次训练数据集
epoch_no = 200
# 最大采样数 --> 限制了每次采样操作最多可以采集到的样本数量。
max_samp = 40
# 正则化参数lambda_1 --> 避免过拟合
lambda_1 = args.lambda1
# 正则化参数lambda_2
lambda_2 = args.lambda2
# Dropout概率 --> dropout参数指定了在每次训练迭代时，神经网络中每个神经元被随机 “丢弃”（即其输出被设置为 0）的概率。
# 增加模型鲁棒性，也防止过拟合。
dropout = args.dropout
# 学习率 --> 决定了每次根据损失函数的梯度来更新模型参数时，参数变化的幅度大小。
lr = args.lr
# 权重衰减系数 --> 模型训练过程中，它会使得模型的权重在每次更新时，除了根据损失函数的梯度进行正常的更新外，
# 还会朝着使权重变小的方向有一定的偏移。防止过拟合,稳定模型训练
decay = args.decay
# SVD分解的秩 --> 通过控制保留的奇异值数量，可以对矩阵进行降维处理或提取其主要特征。
svd_q = args.q

# path = 'data/' + args.data + '/'
# f = open(path+'trnMat.pkl','rb')
# train = pickle.load(f)
# train_csr = (train!=0).astype(np.float32)
# f = open(path+'tstMat.pkl','rb')
# test = pickle.load(f)


# 二、加载数据及预处理；
# load data
import numpy as np
import scipy.sparse as sp

# d = 64  # 嵌入维度
# l = 2  # GNN 层数
# temp = 0.1  # 温度参数
# batch_user = 32  # 用户批次大小
# epoch_no = 100  # 训练轮数
# max_samp = 40  # 最大采样数
# lambda_1 = 0.01  # 正则化参数
# lambda_2 = 1e-5  # 正则化参数
# dropout = 0.2  # Dropout 概率
# lr = 0.001  # 学习率
# decay = 1e-5  # 权重衰减系数
# svd_q = 10  # SVD 分解的秩


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
train_path = 'data/rtrain_0.txt'
train, train_csr = read_file_to_sparse_matrix(train_path)
test_path = 'data/rtest_0.txt'
test, test_csr = read_file_to_sparse_matrix(test_path)

print('Data loaded.')

print('user_num:',train.shape[0],'item_num:',train.shape[1],'lambda_1:',lambda_1,'lambda_2:',lambda_2,'temp:',temp,'q:',svd_q)

epoch_user = min(train.shape[0], 30000)

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
print('Performing SVD...')
svd_u,s,svd_v = torch.svd_lowrank(adj, q=svd_q)
u_mul_s = svd_u @ (torch.diag(s))
v_mul_s = svd_v @ (torch.diag(s))
del s
print('SVD done.')

# process test set
# test_labels = [[] for i in range(test.shape[0])]
# for i in range(len(test.data)):
#     row = test.row[i]
#     col = test.col[i]
#     test_labels[row].append(col)
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
recall_20_x = []
recall_20_y = []
ndcg_20_y = []
recall_40_y = []
ndcg_40_y = []

model = LightGCL(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device)
#model.load_state_dict(torch.load('saved_model.pt'))
optimizer = torch.optim.Adam(model.parameters(),weight_decay=0,lr=lr)
#optimizer.load_state_dict(torch.load('saved_optim.pt'))

current_lr = lr

for epoch in range(epoch_no):
    if (epoch+1)%50 == 0:
        torch.save(model.state_dict(),'saved_model/saved_model_epoch_'+str(epoch)+'.pt')
        torch.save(optimizer.state_dict(),'saved_model/saved_optim_epoch_'+str(epoch)+'.pt')

    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    train_loader.dataset.neg_sampling()
    for i, batch in enumerate(tqdm(train_loader)):
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
        #print('batch',batch)
        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()

        #print(i, len(train_loader), end='\r')

    batch_no = len(train_loader)
    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    epoch_loss_s = epoch_loss_s/batch_no
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)
    print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_r:',epoch_loss_r,'Loss_s:',epoch_loss_s)

    if epoch % 3 == 0:  # test every 10 epochs
        test_uids = np.array([i for i in range(adj_norm.shape[0])])
        batch_no = int(np.ceil(len(test_uids)/batch_user))

        all_recall_20 = 0
        all_ndcg_20 = 0
        all_recall_40 = 0
        all_ndcg_40 = 0
        for batch in tqdm(range(batch_no)):
            start = batch*batch_user
            end = min((batch+1)*batch_user,len(test_uids))

            test_uids_input = torch.LongTensor(test_uids[start:end])
            predictions = model(test_uids_input,None,None,None,test=True)
            predictions = np.array(predictions.cpu())

            #top@20
            recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
            #top@40
            recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

            all_recall_20+=recall_20
            all_ndcg_20+=ndcg_20
            all_recall_40+=recall_40
            all_ndcg_40+=ndcg_40
            #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
        print('-------------------------------------------')
        print('Test of epoch',epoch,':','Recall@20:',all_recall_20/batch_no,'Ndcg@20:',all_ndcg_20/batch_no,'Recall@40:',all_recall_40/batch_no,'Ndcg@40:',all_ndcg_40/batch_no)
        recall_20_x.append(epoch)
        recall_20_y.append(all_recall_20/batch_no)
        ndcg_20_y.append(all_ndcg_20/batch_no)
        recall_40_y.append(all_recall_40/batch_no)
        ndcg_40_y.append(all_ndcg_40/batch_no)

# final test
test_uids = np.array([i for i in range(adj_norm.shape[0])])
batch_no = int(np.ceil(len(test_uids)/batch_user))

all_recall_20 = 0
all_ndcg_20 = 0
all_recall_40 = 0
all_ndcg_40 = 0
for batch in range(batch_no):
    start = batch*batch_user
    end = min((batch+1)*batch_user,len(test_uids))

    test_uids_input = torch.LongTensor(test_uids[start:end])
    predictions = model(test_uids_input,None,None,None,test=True)
    predictions = np.array(predictions.cpu())

    #top@20
    recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
    #top@40
    recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

    all_recall_20+=recall_20
    all_ndcg_20+=ndcg_20
    all_recall_40+=recall_40
    all_ndcg_40+=ndcg_40
    #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
print('-------------------------------------------')
print('Final test:','Recall@20:',all_recall_20/batch_no,'Ndcg@20:',all_ndcg_20/batch_no,'Recall@40:',all_recall_40/batch_no,'Ndcg@40:',all_ndcg_40/batch_no)

recall_20_x.append('Final')
recall_20_y.append(all_recall_20/batch_no)
ndcg_20_y.append(all_ndcg_20/batch_no)
recall_40_y.append(all_recall_40/batch_no)
ndcg_40_y.append(all_ndcg_40/batch_no)

metric = pd.DataFrame({
    'epoch':recall_20_x,
    'recall@20':recall_20_y,
    'ndcg@20':ndcg_20_y,
    'recall@40':recall_40_y,
    'ndcg@40':ndcg_40_y
})
current_t = time.gmtime()
metric.to_csv('log/result_'+args.data+'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.csv')

torch.save(model.state_dict(),'saved_model/saved_model_'+args.data+'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.pt')
torch.save(optimizer.state_dict(),'saved_model/saved_optim_'+args.data+'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.pt')