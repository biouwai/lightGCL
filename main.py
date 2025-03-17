
import logging
import numpy as np
import torch
from model import LightGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor, matrix_decomposition, load_sparse
from scipy.sparse import coo_matrix
import torch.utils.data as data
from utils import TrnData
import warnings
import numpy as np

warnings.filterwarnings("ignore")

logging.basicConfig(filename='ablate.log', level=logging.INFO, format='%(asctime)s - %(message)s')

device = 'cpu'
# 总轮数
total_runs = 1
all_max_auc = []
all_max_aupr = []
all_max_epoch = []

for run in range(total_runs):
    # 实验参数
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
    max_auc=0
    max_aupr=0
    max_epoch=0

    train = load_sparse('dataset/new_train.txt')
    test = load_sparse('dataset/new_test.txt')
    train_csr = train.tocsr()
    origin_train = train.copy()


    # normalizing the adj matrix 利用归一化公式进行矩阵归一化
    rowD = np.array(train.sum(1)).squeeze()
    colD = np.array(train.sum(0)).squeeze()
    for i in range(len(train.data)):
        train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)

    # construct data loader 构造数据加载器
    train = train.tocoo()
    test = test.tocoo()
    train_data = TrnData(train)
    train_loader = data.DataLoader(train_data, batch_size=4096, shuffle=True, num_workers=0)

    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
    adj_norm = adj_norm.coalesce()

    # perform svd reconstruction
    adj = scipy_sparse_mat_to_torch_sparse_tensor(origin_train).coalesce()
    # 将稀疏张量 adj 转换为密集张量（普通的二维数组）。
    adj_dense = adj.to_dense()

    # processed_train = torch.load('processd/one_only1.pt')
    processed_train = torch.load('processd/two_all1.pt')
    # processed_train = torch.load('processd/one_half1.pt')

    svd_u,s,svd_v = matrix_decomposition(processed_train, method='full_svd', q=svd_q)
    u_mul_s = svd_u @ (torch.diag(s))
    v_mul_s = svd_v @ (torch.diag(s))
    del s
    print('SVD done.')

    # process test set -> test_labels: 每个用户的正样本物品 ID 列表。
    max_user_id = max(train.shape[0], test.shape[0])
    test_labels = [[] for i in range(max_user_id)]
    for i in range(len(test.data)):
        row = test.row[i]
        col = test.col[i]
        test_labels[row].append(col)

    model = LightGCL(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, device)
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=0,lr=lr)

    for epoch in range(epoch_no):
        # if (epoch+1)%50 == 0:
        #     torch.save(model.state_dict(),'saved_model/saved_model_epoch_'+str(epoch)+'.pt')
        #     torch.save(optimizer.state_dict(),'saved_model/saved_optim_epoch_'+str(epoch)+'.pt')
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

        if epoch % 5 == 0:  # test every 5 epochs
            with torch.no_grad():  
                # 生成一个包含所有用户 ID 的数组。
                test_uids = np.array([i for i in range(adj_norm.shape[0])])
                # 计算测试数据需要分成多少个批次。
                test_uids_input = torch.LongTensor(test_uids)
                predictions = model(test_uids_input, None, None, None, test=True)
                auc,aupr = metrics(test_uids,predictions,test_labels)
                if (auc + aupr) > (max_auc+max_aupr):
                    max_auc = auc   
                    max_aupr = aupr
                    max_epoch = epoch
                print('auc:',auc,'aupr:',aupr,'epoch:',epoch)
    print('max_auc:',max_auc,'max_aupr:',max_aupr,'max_epoch:',max_epoch)

    all_max_auc.append(max_auc)
    all_max_aupr.append(max_aupr)
    all_max_epoch.append(max_epoch)

# 计算平均值
avg_max_auc = np.mean(all_max_auc)
avg_max_aupr = np.mean(all_max_aupr)
avg_max_epoch = np.mean(all_max_epoch)

print(f"Average max_auc: {avg_max_auc}")
print(f"Average max_aupr: {avg_max_aupr}")
print(f"Average max_epoch: {avg_max_epoch}")

logging.info(' current_strategy: normal, AUC: %s, AUPR: %s,', avg_max_auc, avg_max_aupr)

# # (替换数据集-所有负样本都用预测值)------------------------------------------------------------------------------------
# # 利用模型预测的结果替换原始矩阵中负样本位置的值，同时保持正样本位置不变。
# # 生成所有用户 ID 并使用模型预测所有用户-物品对的概率。
# # 创建原始训练矩阵的副本 ，以便在副本上进行修改。
# # 区分正样本和负样本位置 ，仅更新负样本位置的值。
# # 保存更新后的矩阵 ，并验证保存格式的正确性。
# test_uids = np.arange(adj_norm.shape[0])
# test_uids_input = torch.LongTensor(test_uids)
# predictions = model(test_uids_input, None, None, None, test=True).detach()
# auc,aupr = metrics(test_uids,predictions,test_labels)
# print('finnal_auc:',auc,'finnal_aucr:',aupr,'finnal_epoch:',epoch)
# # 创建原始训练矩阵的副本
# updated_adj_dense = adj_dense.clone()
# # 创建布尔掩码（原始正样本位置为False）
# positive_mask = (adj_dense != 0)
# negative_mask = ~positive_mask
# # 仅替换负样本位置的预测值（保持正样本不变）
# updated_adj_dense[negative_mask] = predictions[negative_mask]
# # 保存更新后的矩阵
# torch.save(updated_adj_dense, 'processd/one_all1.pt')
# print("训练集已更新并保存为 processd/one_all1.pt")
# # 验证保存的格式
# loaded_adj = torch.load('processd/one_all1.pt')
# print("验证保存格式:", 
#       "形状一致" if loaded_adj.shape == adj_dense.shape else "形状不一致",
#       "类型一致" if loaded_adj.dtype == adj_dense.dtype else "类型不一致")

# # (替换数据集-测试集正样本+同等数量负样本用预测值)------------------------------------------------------------------------------------
# # 获取测试集正样本坐标
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
# batch_size = 4096
# predictions = []
# with torch.no_grad():
#     for i in range(0, len(update_users), batch_size):
#         batch_users = torch.LongTensor(update_users[i:i+batch_size])
#         batch_items = torch.LongTensor(update_items[i:i+batch_size])
#         batch_pred = model.get_pair_predictions(batch_users, batch_items)
#         # auc,aupr = metrics(batch_users,predictions,test_labels)
#         # print('finnal2_auc:',auc,'finnal2_aucr:',aupr,'finnal2_epoch:',epoch)
#         predictions.append(batch_pred)

# # 拼接预测结果并验证形状
# predictions = torch.cat(predictions)
# assert predictions.shape == torch.Size([len(update_users)]), \
#     f"预测形状应为{[len(update_users)]}，实际为{predictions.shape}"
# # 更新矩阵
# updated_adj_dense = adj_dense.clone()
# updated_adj_dense[update_users, update_items] = predictions
# # 保存更新后的矩阵
# torch.save(updated_adj_dense, 'processd/one_half1.pt')

# # 验证输出
# print(f"\n更新完成:")
# print(f"- 更新位置总数: {len(update_users)}")
# print(f"- 测试正样本更新: {len(test_pos_users)}")
# print(f"- 随机负样本更新: {len(random_users)}")
# print(f"- 新矩阵非零值: {(updated_adj_dense > 0).sum().item()}")
# print(f"- 保存文件格式: {updated_adj_dense.shape} {updated_adj_dense.dtype}")

# # (替换数据集-测试集正样本+同等数量负样本用预测值)------------------------------------------------------------------------------------
# # 获取测试集正样本坐标
# test_pos_coo = test.tocoo()
# test_pos_users = test_pos_coo.row.astype(np.int32)
# test_pos_items = test_pos_coo.col.astype(np.int32)

# # 使用模型预测测试集正样本的得分
# batch_size = 4096
# predictions = []

# with torch.no_grad():
#     for i in range(0, len(test_pos_users), batch_size):
#         batch_users = torch.LongTensor(test_pos_users[i:i+batch_size])
#         batch_items = torch.LongTensor(test_pos_items[i:i+batch_size])
#         batch_pred = model.get_pair_predictions(batch_users, batch_items)
#         predictions.append(batch_pred)

# # 拼接预测结果
# predictions = torch.cat(predictions)  # 确保 predictions 是 PyTorch 张量

# # 验证预测结果形状
# assert len(predictions) == len(test_pos_users), \
#     f"预测形状应为{len(test_pos_users)}，实际为{len(predictions)}"

# # 更新 adj_dense 中的正样本位置
# updated_adj_dense = adj_dense.clone()
# updated_adj_dense[test_pos_users, test_pos_items] = predictions  # 直接赋值

# # 保存更新后的密集矩阵
# torch.save(updated_adj_dense, 'processd/one_only1.pt')

# # 输出更新信息
# print(f"\n更新完成:")
# print(f"- 测试正样本数量: {len(test_pos_users)}")
# print(f"- 新矩阵非零值数量: {(updated_adj_dense > 0).sum().item()}")
# print(f"- 保存文件格式: {updated_adj_dense.shape} {updated_adj_dense.dtype}")