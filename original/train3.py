import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score
import torch_geometric.nn as pyg_nn
import torch.nn.init as init

# 设置随机种子
SEED = 666
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 定义模型
class GNNDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNDecoder, self).__init__()
        self.gcn1 = pyg_nn.SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.gcn2 = pyg_nn.SAGEConv(hidden_channels, out_channels)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
                # 图编码器
        self.gcn1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.gcn2 = pyg_nn.GCNConv(hidden_channels, out_channels)
        # self.gcn1 = pyg_nn.GATConv(in_channels, hidden_channels,heads=2)
        # self.gcn2 = pyg_nn.GATConv(2*hidden_channels, out_channels,heads=2)
        # self.gcn1 = pyg_nn.SAGEConv(in_channels, hidden_channels)
        # self.gcn2 = pyg_nn.SAGEConv(hidden_channels, out_channels)
        #self.gcn1 = pyg_nn.LEConv(in_channels, hidden_channels)
        #self.gcn2 = pyg_nn.LEConv(hidden_channels, out_channels)
        # self.gcn1 = pyg_nn.GeneralConv(in_channels, hidden_channels)
        # self.gcn2 = pyg_nn.GeneralConv(hidden_channels, out_channels)
        # self.gcn1 = pyg_nn.GraphConv(in_channels, hidden_channels)
        # self.gcn2 = pyg_nn.GraphConv(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for module in [self.gcn1, self.gcn2]:
            for param in module.parameters():
                if param.dim() > 1:
                    init.xavier_uniform_(param)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        RNA_emb = x[:num_RNA]
        drug_emb = x[num_RNA:]
        return RNA_emb, drug_emb

def compute_scores(RNA_emb, drug_emb, edge_index):
    row, col = edge_index
    # 确保 RNA 索引在 [0, num_RNA-1] 范围内
    if row.max() >= num_RNA:
        raise ValueError("RNA index exceeds num_RNA!")
    # 确保药物索引在 [num_RNA, num_RNA + num_drug -1] 范围内
    if col.min() < num_RNA or col.max() >= (num_RNA + num_drug):
        raise ValueError("Drug index out of bounds!")
    
    RNA_features = RNA_emb[row]
    drug_features = drug_emb[col - num_RNA]  # 调整药物索引到 [0, num_drug-1]
    scores = (RNA_features * drug_features).sum(dim=1)
    return scores

def train(model, data, optimizer, train_edge_label_index, train_edge_label):
    model.train()
    optimizer.zero_grad()
    RNA_emb, drug_emb = model(data.x, data.edge_index)
    out = compute_scores(RNA_emb, drug_emb, train_edge_label_index)
    out = torch.sigmoid(out)
    loss = nn.BCELoss()(out, train_edge_label)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    loss.backward()
    optimizer.step()
    return loss.item()

# def evaluate(model, data, test_edge_label_index, test_edge_label):
#     model.eval()
#     with torch.no_grad():
#         RNA_emb, drug_emb = model(data.x, data.edge_index)
#         out = compute_scores(RNA_emb, drug_emb, test_edge_label_index)
#         print(out,'out')
#         y_true = test_edge_label.cpu().numpy()
#         y_scores = torch.sigmoid(out).cpu().numpy()

#                 # 创建全零矩阵，并填充预测得分
#         pred_matrix = torch.zeros((num_RNA, num_drug))
#         rows = test_edge_label_index[0].cpu().numpy()
#         cols = (test_edge_label_index[1] - num_RNA).cpu().numpy()  # 还原药物原始索引
#         pred_matrix[rows, cols] = torch.tensor(y_scores)  # 填充预测值
#         print(pred_matrix,'pred_matrix')
        
#         # 保存预测矩阵
#         torch.save(pred_matrix, "saved_train_dataset/train2.pt")

#         if len(y_true) == 0 or len(y_scores) == 0:
#             raise ValueError("Empty data in evaluation phase!")
#         auc = roc_auc_score(y_true, y_scores)
#         aupr = average_precision_score(y_true, y_scores)
#     return auc, aupr

def evaluate(model, data, test_edge_label_index, test_edge_label):
    model.eval()
    with torch.no_grad():
        RNA_emb, drug_emb = model(data.x, data.edge_index)
        
        # === 生成所有RNA-药物对的预测概率 ===
        # 计算所有可能的分数
        scores = RNA_emb @ drug_emb.T  # 形状: [num_RNA, num_drug]
        full_pred_matrix = torch.sigmoid(scores)  # 转换为概率
        
        # === 保存完整预测矩阵 ===
        torch.save(full_pred_matrix, "saved_train_dataset/full1.pt")
        print("完整预测矩阵已保存")

        # === 原有测试集评估逻辑 ===
        out = compute_scores(RNA_emb, drug_emb, test_edge_label_index)
        y_true = test_edge_label.cpu().numpy()
        y_scores = torch.sigmoid(out).cpu().numpy()
        auc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)
    return auc, aupr

# 数据加载
num_RNA = 1927
num_drug = 216
num_nodes = num_RNA + num_drug

# 构建节点特征（随机初始化）
embedding_dim = 128
RNA_features = torch.randn(num_RNA, embedding_dim)
drug_features = torch.randn(num_drug, embedding_dim)
node_features = torch.cat([RNA_features, drug_features], dim=0)

# 加载边索引（忽略第三列）
def load_edge_index_from_txt(filename):
    rows, cols = [], []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                row, col = map(int, parts[:2])
                rows.append(row)
                cols.append(col)
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError(f"File {filename} is empty or format is invalid.")
    return torch.tensor([rows, cols], dtype=torch.long)

# 加载标签文件（三列）
def load_edge_label_from_txt(filename):
    rows, cols, labels = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                row, col, label = map(int, parts)
                rows.append(row)
                cols.append(col)
                labels.append(label)
    if len(rows) == 0 or len(cols) == 0 or len(labels) == 0:
        raise ValueError(f"File {filename} is empty or format is invalid.")
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)
    return edge_index, labels

# 加载边索引（正样本）
edge_index_train = load_edge_index_from_txt("dataset/rrtest_x copy.txt")  # 三列文件，但只取前两列
edge_index_train[1] += num_RNA  # 调整药物索引
edge_index_test = load_edge_index_from_txt("dataset/rrtest_x copy.txt")    # 测试集边索引
edge_index_test[1] += num_RNA

# 生成反向边（无向图）
reverse_edge_index_train = edge_index_train.flip(0)
reverse_edge_index_test = edge_index_test.flip(0)

# 合并正向边和反向边
edge_index_combined_train = torch.cat([edge_index_train, reverse_edge_index_train], dim=1)
edge_index_combined_test = torch.cat([edge_index_test, reverse_edge_index_test], dim=1)

# 加载标签文件（正负样本）
train_edge_label_index, train_edge_label = load_edge_label_from_txt("dataset/rrtrain_x_n.txt")
test_edge_label_index, test_edge_label = load_edge_label_from_txt("dataset/rrtest_x_n.txt")

# 调整标签文件中的药物索引（关键步骤！）
train_edge_label_index[1] += num_RNA
test_edge_label_index[1] += num_RNA

# 检查边索引是否超出范围
def check_edge_index(edge_index, num_nodes):
    if edge_index.numel() == 0:
        print("Warning: Edge index is empty. Skipping check...")
        return
    if edge_index.max() >= num_nodes:
        raise ValueError(f"Edge index exceeds num_nodes ({num_nodes})")

check_edge_index(edge_index_combined_train, num_nodes)
check_edge_index(edge_index_combined_test, num_nodes)
check_edge_index(train_edge_label_index, num_nodes)
check_edge_index(test_edge_label_index, num_nodes)

# 构建图数据对象
data_train = Data(x=node_features, edge_index=edge_index_combined_train)
data_test = Data(x=node_features, edge_index=edge_index_combined_test)

# 定义模型、优化器
model = GNNDecoder(in_channels=embedding_dim, hidden_channels=512, out_channels=512)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
# optimizer = optim.RMSprop(model.parameters(), lr=0.00001, alpha=0.99)

# 训练模型
num_epochs = 400
for epoch in range(num_epochs):
    loss = train(model, data_train, optimizer, train_edge_label_index, train_edge_label)
    if torch.isnan(torch.tensor(loss)):
        print("Loss is NaN! Stopping training...")
        break
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# 测试模型性能
auc, aupr = evaluate(model, data_test, test_edge_label_index, test_edge_label)
print(f"Test AUC: {auc}, AUPR: {aupr}")