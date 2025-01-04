import torch
import torch.nn as nn
from utils import sparse_dropout, spmm
import torch.nn.functional as F


class LightGCL(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device):
        # 调用父类构造函数，以正确初始化继承自nn.Module的模块相关属性和行为，nn.Module神经网络的一个核心基类
        super(LightGCL,self).__init__()
        # nn.Parameter 用于表示神经网络中的可学习参数，在模型训练时这些参数会根据计算得到的梯度进行更新；
        # nn.init.xavier_uniform_：这是 PyTorch 提供的一种参数初始化方法，即 Xavier 均匀初始化。
        # 其基本思想是根据输入和输出神经元的数量来合理地初始化权重
        # # 创建两个可学习的参数E_u_0和E_i_0，它们分别用于表示用户和物品的初始嵌入向量。
        # # 它们内部的数据元素都是符合 Xavier 均匀分布的随机数值，形状为（n_u,d）
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))

        # 用于存储训练数据相关的稀疏矩阵表示
        self.train_csr = train_csr
        # 存储经过归一化处理后的邻接矩阵,
        # 领接矩阵是描述图结构的关键元素，它定义了节点之间的连接情况
        self.adj_norm = adj_norm
        # self.l表示图神经网络的层数，指定了模型中信息传播的深度。
        self.l = l

        # 创建两个长度为l + 1的列表self.E_u_list和self.E_i_list，
        # 通过 [None] * (l + 1) 的方式初始化列表的长度为 l + 1，其中 l 是之前定义的
        # 代表图神经网络层数的变量。列表中的每个元素初始都被设置为 None，
        # 用于存储每一层的用户和物品嵌入向量。
        # 把初始的用户和物品嵌入向量（E_u_0和E_i_0）分别放入这两个列表的第 0 个位置
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0

        # E代表嵌入向量，包含结点的语义信息、Z代表中间结果、G代表SVD相关传播
        # 同样创建了四个长度为l + 1的列表，分别用于在不同计算阶段
        #（如图神经网络传播、基于svd_adj的传播等）存储中间结果
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0

        # 当 lambda_1 的值较大时，意味着对比学习损失在总损失中占比更大，
        # 模型在训练时会更注重优化对比学习相关的部分，也就是更侧重于
        # 通过对比正负样本去学习用户和物品嵌入向量之间更好的表示关系，
        # 以提升模型在区分不同样本方面的能力。

        # 当 lambda_2 值增大时，正则化损失的影响变大，
        # 模型训练时会更倾向于使参数保持相对较小的值，
        # 促使模型学习到更具泛化能力的模式。
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout

        # 定义了激活函数为LeakyReLU，其负半轴斜率为 0.5
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user

        # 初始化这两个属性为None，后续会用于存储经过多层聚合
        # 等操作后的最终用户和物品嵌入向量，在前向传播过程中逐步更新它们的值。
        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.device = device

    def forward(self, uids, iids, pos, neg, test=False):
        # 测试的分
        if test==True:  # testing phase
            # 计算预测得分 
            # 从所有用户的嵌入向量集合self.E_u中取出对应uids所指定的那些用户的嵌入向量。
            # 然后与self.E_i.T（即物品嵌入向量的转置）进行矩阵乘法操作
            # 计算了每个用户与所有物品之间的某种关联程度或者匹配得分
            preds = self.E_u[uids] @ self.E_i.T
            # 构造一个掩码（mask），用于处理训练集中已经存在交互的情况，
            # 避免在预测时推荐用户已经有过交互的物品
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            # 转换回torch.Tensor类型，使其能够在后续与其他张量进行兼容的运算。
            mask = torch.Tensor(mask)
            # 排除了已交互物品在本次推荐中的影响。而对于掩码中值
            # 为 0 的位置（未交互过的物品），则正常保留其预测得分
            preds = preds * (1-mask) - 1e8 * mask
            # 最终的predictions结果包含了每个用户的推荐物品的索引降序排序信息，
            predictions = preds.argsort(descending=True)
            return predictions
        else:  # training phase
            for layer in range(1,self.l+1):
                # GNN propagation
                # # 用户结点表示更新
                # torch.spmm将将经过 sparse_dropout（随机丢弃） 处理后的邻接矩阵（即上一层物品节点的向量表示）
                # 进行乘法运算。相当于根据图中节点之间的连接关系以及上一层物品节点的特征表示，来更新当前层的用户节点表示
                # 结果存储在 self.Z_u_list[layer] 中
                self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1]))
                # # 物品结点表示更新
                # 调用 .transpose(0, 1) 方法对其进行转置。转置邻接矩阵的目的是因为在图结构中，从用户节点到物品节点的传播方向与从物品节点到用户节点
                # 的传播方向所对应的邻接矩阵结构是不同的（一般来说，邻接矩阵中如果 A[i][j] 表示从节点 i 到节点 j 的连接关系，
                # 那么转置后 A.T[i][j] 就表示从节点 j 到节点 i 的连接关系）
                self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1]))

                # 奇异值分解相关传播
                # svd_adj propagation
                # （GNN）主要关注局部邻域的信息传播，而通过 SVD 的方式可以捕捉更广泛的全局协作关系。
                # self.vt 是与奇异值分解相关的矩阵（在模型初始化时传入），它与上一层的物品节点嵌入向量 self.E_i_list[layer - 1] 相乘，得到中间结果 vt_ei
                vt_ei = self.vt @ self.E_i_list[layer-1]
                # 将其与 vt_ei 相乘，更新当前层在奇异值分解相关传播路径下的用户节点表示，并将结果存储在 self.G_u_list[layer] 中
                self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer-1]
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
            G_u_norm = self.G_u
            E_u_norm = self.E_u
            G_i_norm = self.G_i
            E_i_norm = self.E_i
            
            # 负样本得分计算
            # G_u_norm[uids] @ E_u_norm.T：这部分代码计算了用户嵌入 G_u_norm 和所有用户嵌入 E_u_norm 的转置之间的矩阵乘法，
            # 得到一个形状为 [batch_size, num_users] 的矩阵，表示每个用户与所有用户的相似度。
            # .sum(1)：沿着第1维（即对每个用户的所有其他用户的相似度）求和，得到每个用户与其他所有用户的累积相似度
            # .mean()取平均
            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
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
            loss = loss_r + self.lambda_1 * loss_s + loss_reg
            #print('loss',loss.item(),'loss_r',loss_r.item(),'loss_s',loss_s.item())
            return loss, loss_r, self.lambda_1 * loss_s
