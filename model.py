import torch
import torch.nn as nn
from utils import sparse_dropout
import torch.nn.functional as F


class LightGCL(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device):
        super(LightGCL, self).__init__()

        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))

        # 用于存储训练数据相关的稀疏矩阵表示
        self.train_csr = train_csr
        # 存储经过归一化处理后的邻接矩阵,
        # 邻接矩阵是描述图结构的关键元素，它定义了节点之间的连接情况
        self.adj_norm = adj_norm
        # self.l表示图神经网络的层数，指定了模型中信息传播的深度。
        self.l = l

        self.E_u_list = [None] * (l + 1)
        self.E_i_list = [None] * (l + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0

        self.Z_u_list = [None] * (l + 1)
        self.Z_i_list = [None] * (l + 1)
        self.G_u_list = [None] * (l + 1)
        self.G_i_list = [None] * (l + 1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0

        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout

        # 定义了激活函数为LeakyReLU，其负半轴斜率为 0.5
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user

        # 等操作后的最终用户和物品嵌入向量，在前向传播过程中逐步更新它们的值。
        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.device = device

    def forward(self, uids, iids, pos, neg, test=False):
        # 测试部分
        if test == True:  # testing phase
            # 计算预测得分 
            predictions = self.E_u[uids] @ self.E_i.T
            # 使用 sigmoid 函数将得分转换为概率值，表示用户会产生目标行为的概率
            predictions = torch.sigmoid(predictions)
            return predictions
        else:  # training phase
            for layer in range(1, self.l + 1):
                # 结果存储在 self.Z_u_list[layer] 中
                self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm, self.dropout), self.E_i_list[layer - 1]))
                # # 物品结点表示更新
                self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1), self.E_u_list[layer - 1]))

                vt_ei = self.vt @ self.E_i_list[layer - 1]
                
                self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer - 1]
                self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

                # 聚合操作
                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer]
                self.E_i_list[layer] = self.Z_i_list[layer]

            # 得到综合了各层信息的用户和物品表示， 跨层聚合
            self.G_u = sum(self.G_u_list)
            self.G_i = sum(self.G_i_list)

            # aggregate across layers
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            # 对比学习损失计算
            # cl loss
            # G_u_norm = self.G_u
            # E_u_norm = self.E_u
            # G_i_norm = self.G_i
            # E_i_norm = self.E_i

            G_u_norm = F.normalize(self.G_u, p=2, dim=1)
            E_u_norm = F.normalize(self.E_u, p=2, dim=1)

            G_i_norm = F.normalize(self.G_i, p=2, dim=1)
            E_i_norm = F.normalize(self.E_i, p=2, dim=1)

            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp, -5.0, 5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp, -5.0, 5.0)).mean()
            # 计算对比学习损失 
            loss_s = -pos_score + neg_score

            # bpr loss
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # 正则化损失（reg loss）计算
            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            # 总损失计算与返回
            # total loss
            loss = loss_r + loss_reg + self.lambda_1 * loss_s
            # loss = loss_r + loss_reg
            return loss, loss_r, self.lambda_1 * loss_s