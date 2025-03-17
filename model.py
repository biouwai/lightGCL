import torch
import torch.nn as nn
from utils import sparse_dropout
import torch.nn.functional as F


class LightGCL(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, device):
        super(LightGCL, self).__init__()
        # 使用Xavier方法初始化RNA与药物的嵌入嵌入向量
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))

        # 用于存储训练数据相关的稀疏矩阵表示
        self.train_csr = train_csr
        # 存储经过归一化处理后的邻接矩阵,邻接矩阵是描述图结构的关键元素，它定义了节点之间的连接情况
        self.adj_norm = adj_norm
        # self.l表示图神经网络的层数，指定了模型中信息传播的深度。
        self.l = l

        # 初始化各层的用户和物品嵌入列表
        # self.E_u_list ：综合局部和全局信息，形成最终的用户嵌入表示。
        self.E_u_list = [None] * (l + 1)
        self.E_i_list = [None] * (l + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        # self.Z_u_list ：通过邻接矩阵传播捕获局部关系。
        self.Z_u_list = [None] * (l + 1)
        self.Z_i_list = [None] * (l + 1)
        # self.G_u_list ：通过svd分解捕获全局结构信息。
        self.G_u_list = [None] * (l + 1)
        self.G_i_list = [None] * (l + 1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0

        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout

        # 等操作后的最终用户和物品嵌入向量，在前向传播过程中逐步更新它们的值。
        self.E_u = None
        self.E_i = None

        # SVD 分解的结果，用于增强全局信息
        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.device = device

    def forward(self, uids, iids, pos, neg, test=False):
        # 测试部分
        if test == True: 
            # 根据用户和物品的嵌入表示，计算RNA对所有药物耐药性的预测得分
            predictions = self.E_u[uids] @ self.E_i.T
            # 使用 sigmoid 函数将得分转换为概率值，表示RNA与药物有关联耐药性的概率
            predictions = torch.sigmoid(predictions)
            return predictions
        else:  
            for layer in range(1, self.l + 1):
                # 通过邻接矩阵传播捕获局部关系
                # 将稀疏矩阵与上一层的物品嵌入相乘，得到当前层的RNA与用户表示
                self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm, self.dropout), self.E_i_list[layer - 1]))
                self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1), self.E_u_list[layer - 1]))
                
                # 使用 SVD 分解结果增强全局信息
                # 结合 vt 和上一层的药物与RNA嵌入，更新RNA与用户的全局表示
                vt_ei = self.vt @ self.E_i_list[layer - 1]
                self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer - 1]
                self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

                # 聚合操作
                # 目前直接将局部信息赋值给最终嵌入
                self.E_u_list[layer] = self.Z_u_list[layer]
                self.E_i_list[layer] = self.Z_i_list[layer]
                # 如果希望模型自适应地学习局部和全局信息的重要性，可以通过动态调整权重的方式。可以增加指标
                # alpha = torch.sigmoid(torch.randn(1))  # 动态权重
                # self.E_u_list[layer] = alpha * self.Z_u_list[layer] + (1 - alpha) * self.G_u_list[layer]
                # self.E_i_list[layer] = alpha * self.Z_i_list[layer] + (1 - alpha) * self.G_i_list[layer]

            # 得到综合了各层信息的用户和物品表示， 跨层聚合
            self.G_u = sum(self.G_u_list)
            self.G_i = sum(self.G_i_list)

            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            # 对比学习损失计算
            # 对嵌入进行 L2 归一化，确保相似性计算不受向量长度影响
            G_u_norm = F.normalize(self.G_u, p=2, dim=1)
            E_u_norm = F.normalize(self.E_u, p=2, dim=1)
            G_i_norm = F.normalize(self.G_i, p=2, dim=1)
            E_i_norm = F.normalize(self.E_i, p=2, dim=1)

            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp, -5.0, 5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp, -5.0, 5.0)).mean()
            loss_s = -pos_score + neg_score

            # bpr loss
            u_emb = self.E_u[uids]  # 用户嵌入
            pos_emb = self.E_i[pos]  # 正样本物品嵌入
            neg_emb = self.E_i[neg]  # 负样本物品嵌入
            pos_scores = (u_emb * pos_emb).sum(-1)  # 正样本得分
            neg_scores = (u_emb * neg_emb).sum(-1)  # 负样本得分
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # 正则化损失（reg loss）计算
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            # 总损失计算与返回
            loss = loss_r + loss_reg + self.lambda_1 * loss_s
            # loss = loss_r 
            return loss, loss_r, self.lambda_1 * loss_s

    def get_pair_predictions(self, users, items):
        # 获取用户和物品的嵌入向量
        user_embeddings = self.E_u[users]  # 形状: [batch_size, d]
        item_embeddings = self.E_i[items]  # 形状: [batch_size, d]
        # 计算点积作为预测得分
        predictions = (user_embeddings * item_embeddings).sum(dim=1)  # 形状: [batch_size]
        predictions = torch.sigmoid(predictions)
        return predictions