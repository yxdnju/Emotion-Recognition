from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool,global_max_pool
import torch.nn.functional as F
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from DEAP_model.GCN_layer import *
# 两种视图下的图卷积模型，包括自适应邻接矩阵、空间距离矩阵，
# 在此基础上增加 CNN时间卷积层


def optimize_adjacency(features, alpha=0.1, learning_rate=0.001, iterations=100):
    device = features.device
    N, D = features.shape
    A = torch.rand(N, N, device=device)
    A = (A + A.t()) / 2

    for _ in range(iterations):
        X_prime = torch.matmul(A, features)
        grad_reconstruction = 2 * torch.matmul(torch.matmul(A, features) - features, features.t())
        grad_regularization = alpha * torch.sign(A)
        A -= learning_rate * (grad_reconstruction + grad_regularization)
        A = A.clamp(0)
        A = (A + A.t()) / 2

    return A

def compute_pearson_correlation(X):
    X_tensor = X.clone().detach()
    correlation_matrix = torch.corrcoef(X_tensor)
    # print('correlation_matrix',correlation_matrix.shape)

    return correlation_matrix

def adjust_adjacency_matrix(A, correlation_matrix, threshold=0):
    A_tensor = A.clone().detach()
    correlation_matrix_tensor = correlation_matrix.clone().detach()
    A_adjusted = torch.mul(A_tensor, correlation_matrix_tensor)
    A_adjusted[A_adjusted < threshold] = 0
    return A_adjusted

# 假设已经有了Electrodes62类和相关计算距离的函数
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, out, dropout):
#         super(GCN, self).__init__()
#         self.gc1 = GCNConv(nfeat, nhid)
#         self.gc2 = GCNConv(nhid, out)
#         self.dropout = dropout
#
#     def forward(self, x, edge_index):
#         x = F.relu(self.gc1(x, edge_index))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.gc2(x, edge_index)
#         return x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, out)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.gc1(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight)
        return x



class GraphOperations(nn.Module):
    def __init__(self, num_node_features, num_classes, nhid1, nhid2, dropout,A_spatial):
        super(GraphOperations, self).__init__()

        self.A_spatial = A_spatial
        # 特定图卷积网络
        self.SGCN1 = GCN(num_node_features, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(num_node_features, nhid1, nhid2, dropout)

        # 共享图卷积网络
        self.CGCN = GCN(num_node_features, nhid1, nhid2, dropout)

        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # 注意力机制
        self.attention = Attention(nhid2)

        # 最终的分类层
        # self.lin = nn.Linear(nhid2 * 3, num_classes)
        self.MLP = nn.Sequential(
            nn.Linear(nhid2 * 3, nhid2),  # 将堆叠后的维度考虑在内
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid2, num_classes),
            nn.LogSoftmax(dim=1)
        )




    def forward(self, batch_features_np, A_spatial):
        # 处理自适应邻接矩阵和空间邻接矩阵生成的批次数据
        batch_data_adaptive = self.process_batch(batch_features_np)
        batch_data_spatial = self.process_batch_with_spatial(batch_features_np, A_spatial)

        # 提取图数据
        x_adaptive, edge_index_adaptive, edge_weight_adaptive = batch_data_adaptive.x, batch_data_adaptive.edge_index, batch_data_adaptive.edge_attr
        x_spatial, edge_index_spatial, edge_weight_spatial = batch_data_spatial.x, batch_data_spatial.edge_index, batch_data_spatial.edge_attr

        # 使用特定的GCN处理自适应邻接矩阵和空间邻接矩阵，并传入边权重
        emb1 = self.SGCN1(x_adaptive, edge_index_adaptive, edge_weight_adaptive)
        emb2 = self.SGCN2(x_spatial, edge_index_spatial, edge_weight_spatial)

        # 同样，使用共享的GCN处理两种邻接矩阵，并传入边权重
        com1 = self.CGCN(x_adaptive, edge_index_adaptive, edge_weight_adaptive)
        com2 = self.CGCN(x_spatial, edge_index_spatial, edge_weight_spatial)

        # 计算共享图卷积网络输出的平均值
        Xcom = (com1 + com2) / 2

        # 将emb1, emb2和Xcom堆叠，为应用注意力机制做准备
        emb = torch.stack([emb1, emb2, Xcom], dim=1)  # 注意dim=1是堆叠的维度

        # 应用注意力机制
        emb, attention_weights = self.attention(emb.unsqueeze(1))  # 注意输入维度调整

        emb = global_max_pool(emb.view(emb.size(0), -1), batch_data_adaptive.batch)

        out = self.MLP(emb)

        return out


    # def forward(self, batch_features_np, A_spatial):
    #     # 处理自适应邻接矩阵和空间邻接矩阵生成的批次数据
    #     batch_data_adaptive = self.process_batch(batch_features_np)
    #     # print("batch_data_adaptive shape:", batch_data_adaptive.x.shape)
    #     batch_data_spatial = self.process_batch_with_spatial(batch_features_np, A_spatial)
    #     # print("batch_data_spatial shape:", batch_data_spatial.x.shape)
    #
    #     # 提取图数据
    #     x_adaptive, edge_index_adaptive, _ = batch_data_adaptive.x, batch_data_adaptive.edge_index, batch_data_adaptive.batch
    #     x_spatial, edge_index_spatial, _ = batch_data_spatial.x, batch_data_spatial.edge_index, batch_data_spatial.batch
    #     # print("x_adaptive shape:", x_adaptive.shape)
    #     # print("edge_index_adaptive shape:", edge_index_adaptive.shape)
    #     # print("x_spatial shape:", x_spatial.shape)
    #     # print("edge_index_spatial shape:", edge_index_spatial.shape)
    #
    #     # 使用特定的GCN处理自适应邻接矩阵和空间邻接矩阵
    #     emb1 = self.SGCN1(x_adaptive, edge_index_adaptive)
    #     # print("emb1 shape after SGCN1:", emb1.shape)
    #     emb2 = self.SGCN2(x_spatial, edge_index_spatial)
    #     # print("emb2 shape after SGCN2:", emb2.shape)
    #
    #     # 使用共享的GCN同时处理两种邻接矩阵
    #     com1 = self.CGCN(x_adaptive, edge_index_adaptive)
    #     # print("com1 shape after CGCN:", com1.shape)
    #     com2 = self.CGCN(x_spatial, edge_index_spatial)
    #     # print("com2 shape after CGCN:", com2.shape)
    #
    #     # 融合特定和共享图卷积网络的输出
    #     # emb = torch.cat([emb1, emb2, (com1 + com2) / 2], dim=1)
    #
    #     # 计算共享图卷积网络输出的平均值
    #     Xcom = (com1 + com2) / 2
    #
    #     # 将emb1, emb2和Xcom堆叠，为应用注意力机制做准备
    #     emb = torch.stack([emb1, emb2, Xcom], dim=1)  # 注意dim=1是堆叠的维度
    #
    #
    #     # print("emb shape after concatenation:", emb.shape)
    #
    #     # 应用注意力机制
    #     emb, attention_weights = self.attention(emb.unsqueeze(1))  # 注意输入维度调整
    #
    #     # print("emb shape after attention:", emb.shape)
    #     # print("attention_weights shape:", attention_weights.shape)
    #
    #     # 分类
    #
    #     # emb = global_mean_pool(emb.view(emb.size(0), -1), batch_data_adaptive.batch)
    #
    #     # out = self.MLP(emb_pooled)
    #     # emb = emb.view(emb.size(0), -1)  # 展平
    #     # print('展平后',emb.shape)
    #
    #     emb_flattened = emb.view(-1, emb.size(-1))
    #     print("emb shape after  emb_flattened:",  emb_flattened.shape)
    #
    #
    #     emb = global_mean_pool(emb_flattened, batch_data_adaptive.batch)
    #     print("emb shape after global_mean_pool:", emb.shape)
    #
    #
    #     out = self.MLP(emb)
    #
    #     # out = self.lin(emb)
    #
    #     print("out shape after linear and log softmax:", out.shape)
    #
    #     return out

    def process_batch(self, batch_features_np):
        batch_data_list = []
        for features_np in batch_features_np:
            optimized_A = optimize_adjacency(features_np)
            correlation_matrix = compute_pearson_correlation(features_np)
            A_adjusted = adjust_adjacency_matrix(optimized_A, correlation_matrix)
            edge_index = torch.nonzero(A_adjusted, as_tuple=False).t().contiguous()
            edge_weights = A_adjusted[edge_index[0], edge_index[1]]
            edge_weights = edge_weights.view(-1, 1)
            x = features_np.clone().detach().float()
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights)
            batch_data_list.append(data)

        batch_data = Batch.from_data_list(batch_data_list)
        return batch_data


    def process_batch_with_spatial(self, batch_features_np, A_spatial):
        batch_data_list = []
        for features_np in batch_features_np:
            # 这里直接使用A_spatial作为邻接矩阵
            edge_index = torch.nonzero(A_spatial, as_tuple=False).t().contiguous()
            edge_weights = A_spatial[edge_index[0], edge_index[1]]
            # x = torch.tensor(features_np, dtype=torch.float)
            x = features_np.clone().detach().float()  # 假设 features_np 已经是一个张量

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights.view(-1, 1))
            batch_data_list.append(data)
        return Batch.from_data_list(batch_data_list)

