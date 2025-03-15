
import logging
import numpy as np
import torch
from model import LightGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor,matrix_decomposition
import torch.utils.data as data
from utils import TrnData
import warnings
import numpy as np
from scipy.sparse import coo_matrix

warnings.filterwarnings("ignore")

device = 'cpu'

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
    lambda_1=0.005
    lr=0.001
    svd_q=64
    temp=0.1
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
    train = load_sparse('dataset/rtrain_0.txt')
    train_csr = train.tocsr()
    test = load_sparse('dataset/rtest_0.txt')

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
    train_loader = data.DataLoader(train_data, batch_size=4096, shuffle=True, num_workers=0)

    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
    adj_norm = adj_norm.coalesce()

# 下次使用时可以直接加载
    # perform svd reconstruction
    adj = scipy_sparse_mat_to_torch_sparse_tensor(train1).coalesce()
    test1 = scipy_sparse_mat_to_torch_sparse_tensor(test1).coalesce()
    # 4. 转换为密集张量
    train1 = torch.load('saved_train_dataset/all.pt')# 转换为稀疏格式

    adj_dense = adj.to_dense()
    adj_dense1 = test1.to_dense()


    svd_u,s,svd_v = matrix_decomposition(adj_dense, method='full_svd', q=svd_q)
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
            with torch.no_grad():  # 新增的上下文管理器
                test_uids = np.array([i for i in range(adj_norm.shape[0])])
                batch_no = int(np.ceil(len(test_uids)/batch_user))
                test_uids_input = torch.LongTensor(test_uids)
                predictions = model(test_uids_input, None, None, None, test=True)
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
    
# ------------------------------------------------------------------
# 只取对应正样本
# 获取测试集正样本坐标
# test_pos_coo = test.tocoo()
# test_pos_users = test_pos_coo.row.astype(np.int32)
# test_pos_items = test_pos_coo.col.astype(np.int32)

# # 分批预测测试正样本的概率
# batch_size = 2048  # 根据内存调整批大小
# predictions = []
# with torch.no_grad():
#     for i in range(0, len(test_pos_users), batch_size):
#         batch_users = torch.LongTensor(test_pos_users[i:i+batch_size])
#         batch_items = torch.LongTensor(test_pos_items[i:i+batch_size])
#         # 使用模型预测对应位置的分数
#         batch_pred = model.get_pair_predictions(batch_users, batch_items)
#         predictions.append(batch_pred)

# # 拼接预测结果
# predictions = torch.cat(predictions)

# # 验证预测结果的形状
# assert predictions.shape == torch.Size([len(test_pos_users)]), \
#     f"预测形状应为{[len(test_pos_users)]}，实际为{predictions.shape}"

# # 创建更新后的矩阵
# updated_adj_dense = adj_dense.clone()

# # 仅更新测试正样本位置的概率
# updated_adj_dense[test_pos_users, test_pos_items] = predictions

# # 保存更新后的矩阵
# torch.save(updated_adj_dense, 'saved_train_dataset/only-positive.pt')

# # 输出更新信息
# print(f"\n更新完成:")
# print(f"- 更新位置总数: {len(test_pos_users)}")
# print(f"- 测试正样本位置已更新为模型预测概率")
# print(f"- 保存文件格式: {updated_adj_dense.shape} {updated_adj_dense.dtype}")

# ...（前面的训练代码保持不变）------------------------------------------------------------------------------------
# 以下为全部

# 在训练结束后添加以下代码
# 生成所有用户的预测得分
# test_uids = np.arange(adj_norm.shape[0])
# test_uids_input = torch.LongTensor(test_uids)
# predictions = model(test_uids_input, None, None, None, test=True).detach()

# # 创建原始训练矩阵的副本
# updated_adj_dense = adj_dense.clone()

# # 创建布尔掩码（原始正样本位置为False）
# positive_mask = (adj_dense != 0)
# negative_mask = ~positive_mask

# # 仅替换负样本位置的预测值（保持正样本不变）
# updated_adj_dense[negative_mask] = predictions[negative_mask]

# # 保存更新后的矩阵
# torch.save(updated_adj_dense, 'saved_train/all.pt')
# print("训练集已更新并保存为 all.pt")

# # 验证保存的格式
# loaded_adj = torch.load('saved_train/all.pt')
# print("验证保存格式:", 
#       "形状一致" if loaded_adj.shape == adj_dense.shape else "形状不一致",
#       "类型一致" if loaded_adj.dtype == adj_dense.dtype else "类型不一致")

# ================== 训练结束后添加以下代码 ==================
# 获取测试集正样本坐标
# test_pos_coo = test.tocoo()
# test_pos_users = test_pos_coo.row.astype(np.int32)
# test_pos_items = test_pos_coo.col.astype(np.int32)

# # 生成随机负样本坐标（与测试正样本数量相同）
# num_replace = len(test_pos_users)
# np.random.seed(2048)  # 保持可复现性

# # 创建候选负样本池（排除所有已知正样本）
# all_pos_set = set(zip(train.row, train.col)) | set(zip(test_pos_users, test_pos_items))
# valid_negs = []
# max_retry = 5

# # 高效采样逻辑
# for _ in range(max_retry):
#     # 生成候选样本
#     candidate_users = np.random.randint(0, train.shape[0], size=num_replace*10)
#     candidate_items = np.random.randint(0, train.shape[1], size=num_replace*10)
    
#     # 过滤有效负样本
#     for u, i in zip(candidate_users, candidate_items):
#         if (u, i) not in all_pos_set and adj_dense[u, i] == 0:
#             valid_negs.append((u, i))
#         if len(valid_negs) >= num_replace:
#             break
#     if len(valid_negs) >= num_replace:
#         break

# random_users, random_items = zip(*valid_negs[:num_replace])
# random_users = np.array(random_users)
# random_items = np.array(random_items)

# # 合并更新坐标
# update_users = np.concatenate([test_pos_users, random_users])
# update_items = np.concatenate([test_pos_items, random_items])

# # 分批预测防止内存溢出
# batch_size = 2048
# predictions = []
# with torch.no_grad():
#     for i in range(0, len(update_users), batch_size):
#         batch_users = torch.LongTensor(update_users[i:i+batch_size])
#         batch_items = torch.LongTensor(update_items[i:i+batch_size])
#         batch_pred = model.get_pair_predictions(batch_users, batch_items)
#         predictions.append(batch_pred)

# # 拼接预测结果并验证形状
# predictions = torch.cat(predictions)
# assert predictions.shape == torch.Size([len(update_users)]), \
#     f"预测形状应为{[len(update_users)]}，实际为{predictions.shape}"

# # 更新矩阵
# updated_adj_dense = adj_dense.clone()
# updated_adj_dense[update_users, update_items] = predictions

# # 保存更新后的矩阵
# torch.save(updated_adj_dense, 'saved_train_dataset/half.pt')

# # 验证输出
# print(f"\n更新完成:")
# print(f"- 更新位置总数: {len(update_users)}")
# print(f"- 测试正样本更新: {len(test_pos_users)}")
# print(f"- 随机负样本更新: {len(random_users)}")
# print(f"- 新矩阵非零值: {(updated_adj_dense > 0).sum().item()}")
# print(f"- 保存文件格式: {updated_adj_dense.shape} {updated_adj_dense.dtype}")