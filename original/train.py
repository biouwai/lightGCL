import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch_geometric.nn as pyg_nn
import random
import pandas as pd

# %%
SEED = 666

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



class GNNDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNDecoder, self).__init__()
        # 图编码器
        #self.gcn1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        #self.gcn2 = pyg_nn.GCNConv(hidden_channels, out_channels)
        #self.gcn1 = pyg_nn.GATConv(in_channels, hidden_channels,heads=2)
        #self.gcn2 = pyg_nn.GATConv(2*hidden_channels, out_channels,heads=2)
        self.gcn1 = pyg_nn.SAGEConv(in_channels, hidden_channels)
        self.gcn2 = pyg_nn.SAGEConv(hidden_channels, out_channels)
        #self.gcn1 = pyg_nn.LEConv(in_channels, hidden_channels)
        #self.gcn2 = pyg_nn.LEConv(hidden_channels, out_channels)
        #self.gcn1 = pyg_nn.GeneralConv(in_channels, hidden_channels)
        #self.gcn2 = pyg_nn.GeneralConv(hidden_channels, out_channels)
        #self.gcn1 = pyg_nn.GraphConv(in_channels, hidden_channels)
        #self.gcn2 = pyg_nn.GraphConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 编码器：两层GCN
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)

        # 获取RNA和drug的特征（拼接）
        RNA_emb = x[:num_RNA]  # RNA部分
        drug_emb = x[num_RNA:]  # drug部分

        # 解码器：通过点乘
        out = torch.matmul(RNA_emb, drug_emb.T)
        return out


def train(model, data, optimizer, train_edge_label_index, train_edge_label):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # 前向传播

    # 提取正负样本对应的节点对的预测分数
    out = out[train_edge_label_index[0], train_edge_label_index[1]]

    # 标签：正样本为1，负样本为0
    labels = train_edge_label

    # 计算损失
    loss = nn.BCEWithLogitsLoss()(out, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data, test_edge_label_index, test_edge_label):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index) # 将out保存到一个文件,保存成一个矩阵格式 --

        # 获取正负测试样本的输出
        out = out[test_edge_label_index[0], test_edge_label_index[1]] 
        labels = test_edge_label

        # 计算AUC和AUPR
        y_true = labels.cpu().numpy()
        y_scores = out.cpu().numpy()
        df = pd.DataFrame({
            'y_scores': y_scores,  # 第一列：y_scores
            'y_true': y_true  # 第二列：y_true
        })
        # res_name = f'./4.results/{method_name}_{edge_type}_score.csv' # 第二个策略，将预测的测试样本的得分加入到main2的洗脸
        # df.to_csv( res_name, index=False)
        auc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)
    return auc, aupr


def edge_index_to_matrix(edge_index, num_code_1, num_code_2):
    # 创建一个大小为 (num_nodes, num_nodes) 的全零矩阵
    matrix = torch.zeros((num_code_1, num_code_2), dtype=torch.float32)
    # 将边信息转化为邻接矩阵
    row, col = edge_index
    matrix[row, col] = 1  # 对应的边设置为1
    matrix[col, row] = 1  # 如果是无向图，反向边也设置为1
    return matrix


learn_rate=0.0005 # 调整到最优
#edge_type = 'LncDrug'
#node_type = 'lncRNA'
#sim_type = 'LncLnc'
# edge_type = 'MiDrug'
# node_type = 'miRNA'
# sim_type = 'MiMi'
# method_name = 'GCN'

# train_data = torch.load('./3.heter_data/' + edge_type + '_train_data.pth')
# test_data = torch.load('./3.heter_data/' + edge_type + '_test_data.pth')
num_RNA = 1927 # RNA的数量--
num_drug =216    # drug的数量--
num_nodes = num_RNA + num_drug
# RNA_features = train_data[node_type].x
# drug_features = train_data['drug'].x
#将原先的RNA表达特征、药物SMILES特征，分别对RNA_feature和drug_feature取one hot编码
RNA_labels = torch.arange(num_RNA)
RNA_features = torch.nn.functional.one_hot(RNA_labels, num_classes=num_RNA)
drug_labels = torch.arange(num_drug)
drug_features = torch.nn.functional.one_hot(drug_labels, num_classes=num_drug)

# 构建节点特征矩阵 ,以lncRNA-drug为例 (lncRNA + drug) -> (1322 + 219, 938 + 768)
node_features = torch.zeros((num_nodes, RNA_features.shape[1]+drug_features.shape[1]), dtype=torch.float32)
node_features[:num_RNA, :RNA_features.shape[1]] = RNA_features
node_features[num_RNA:, RNA_features.shape[1]:] = drug_features
node_features  = node_features
print('node_features:',node_features.device)

#构建图中的所有边（本工作还包含了RNA直接的相似边和药物直接的相似边，以及RNA-药物交互边）
#由于RNA和药物原先的索引都是从0开始，现在将两者看作是同一类型，需要调整药物的索引
# edge_index_RNA_sim = train_data[node_type,sim_type,node_type].edge_index
# edge_index_drug_sim = train_data['drug','DrugDrug','drug'].edge_index
# edge_index_drug_sim[0] += num_RNA
# edge_index_drug_sim[1] += num_RNA

def load_edge_index_from_txt(filename):
    rows, cols = [], []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                row, col = map(int, parts)
                rows.append(row)
                cols.append(col)
    return torch.tensor([rows, cols], dtype=torch.long)

# 加载 edge_index_train
edge_index_train = load_edge_index_from_txt("dataset/rrtrain_x.txt")  #文件改为2 X 正样本数，一行为RNA，第二行为药物索引-- 
edge_index_train[1] += num_RNA
edge_index_test = load_edge_index_from_txt("dataset/rrtest_x.txt") # 对应的测试文件，这里不用负样本--
edge_index_test[1] += num_RNA 
#由于图为无向图，需要添加反向边
# reverse_edge_index_RNA_sim = edge_index_RNA_sim.flip(0)
# reverse_edge_index_drug_sim = edge_index_drug_sim.flip(0)
reverse_edge_index_train = edge_index_train.flip(0)
reverse_edge_index_test = edge_index_test.flip(0)
#合并正向边和反向边
# edge_index_RNA_sim_combined = torch.cat([edge_index_RNA_sim, reverse_edge_index_RNA_sim], dim=1)
# edge_index_drug_sim_combined = torch.cat([edge_index_drug_sim, reverse_edge_index_drug_sim], dim=1)
#edge_index_combined = torch.cat([edge_index, reverse_edge_index], dim=1)
edge_index_combined_train = torch.cat([edge_index_train, reverse_edge_index_train], dim=1)
edge_index_combined_test = torch.cat([edge_index_test, reverse_edge_index_test], dim=1)

# ...（前面的代码保持不变）

# 加载训练的正负样本边索引并调整药物索引
train_edge_label_index = load_edge_index_from_txt("dataset/rrtrain_x_n.txt")
train_edge_label_index[1] += num_RNA  # 调整药物索引

# 加载测试的正负样本边索引并调整药物索引
test_edge_label_index = load_edge_index_from_txt("dataset/rrtest_x_n.txt")
test_edge_label_index[1] += num_RNA  # 调整药物索引

# 构建训练标签：假设前一半为正样本，后一半为负样本
num_edges = train_edge_label_index.shape[1]
half = num_edges // 2
train_edge_label = torch.cat([torch.ones(half), torch.zeros(num_edges - half)])

# 构建测试标签
num_edges_test = test_edge_label_index.shape[1]
half_test = num_edges_test // 2
test_edge_label = torch.cat([torch.ones(half_test), torch.zeros(num_edges_test - half_test)])

# 确保数据正确性
# print("训练边索引调整后：", train_edge_label_index.max(), "应小于总节点数", num_nodes)
# print("测试边索引调整后：", test_edge_label_index.max(), "应小于总节点数", num_nodes)

# 构建图数据对象
data_train = Data(x=node_features, edge_index=edge_index_combined_train)
data_test = Data(x=node_features, edge_index=edge_index_combined_test)

# 定义模型、优化器
model = GNNDecoder(in_channels=node_features.shape[1], hidden_channels=128, out_channels=128) # 调整到最优 hidden_channels=128, out_channels=128，需要同步数量 --
#optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.RMSprop(model.parameters(), lr=learn_rate, alpha=0.99)

# 添加梯度裁剪并调整学习率
optimizer = optim.RMSprop(model.parameters(), lr=0.0001)  # 降低学习率
# 在训练循环中添加梯度裁剪
def train(model, data, optimizer, train_edge_label_index, train_edge_label):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    out = out[train_edge_label_index[0], train_edge_label_index[1]]
    loss = nn.BCEWithLogitsLoss()(out, train_edge_label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
    optimizer.step()
    return loss.item()

# ...（其余代码保持不变）



# 训练模型
num_epochs = 200 # 调整到最优
for epoch in range(num_epochs):
    loss = train(model, data_train, optimizer, train_edge_label_index, train_edge_label)
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# 测试模型性能
auc, aupr = evaluate(model, data_test, test_edge_label_index, test_edge_label)
print(f'Test AUC: {auc}, AUPR: {aupr}')
