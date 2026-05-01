import torch.nn.functional as F
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from DEAP_model.GCN_layer import *
# 两种视图下的图卷积模型，包括自适应邻接矩阵、空间距离矩阵，
# 在此基础上增加 CNN时间卷积层


def optimize_adjacency(features, alpha=0.3, learning_rate=0.001, iterations=100):
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

def adjust_adjacency_matrix(A, correlation_matrix, threshold=0.2):
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
        w = self.project(z)  # [batch_size, 3, 1]
        beta = torch.softmax(w, dim=1)  # Softmax over the 3 embeddings
        return (beta * z).sum(1), beta  # Weighted sum of embeddings


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()


        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        # print('GCN',x.shape,adj.shape)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GraphOperations1(nn.Module):
    def __init__(self, nfeat, nhid, nclass,  A_spatial,dropout):
        super(GraphOperations1, self).__init__()
        self.dropout = dropout
        self.A_spatial = A_spatial

        # 对自适应邻接矩阵的特定图卷积和公共图卷积
        self.adaptive_SGCN = GCN(nfeat, nhid, nhid, dropout)
        self.adaptive_CGCN = GCN(nfeat, nhid, nhid, dropout)

        # 对空间距离邻接矩阵的特定图卷积和公共图卷积
        self.spatial_SGCN = GCN(nfeat, nhid, nhid, dropout)
        self.spatial_CGCN = GCN(nfeat, nhid, nhid, dropout)

        # 注意力机制
        self.attention = Attention(nhid * 2)

        # 分类器
        self.classifier = nn.Linear(nhid * 2, nclass)

    def forward(self, x, adj_spatial):
        batch_size, num_nodes, num_features = x.size()
        adj_adaptive_batch = torch.zeros(batch_size, num_nodes, num_nodes, device=x.device)

        for idx in range(batch_size):
            # 动态生成自适应邻接矩阵
            features_single = x[idx]  # 移除batch维度进行处理
            optimized_A = optimize_adjacency(features_single)
            correlation_matrix = compute_pearson_correlation(features_single)
            A_adjusted = adjust_adjacency_matrix(optimized_A, correlation_matrix)
            adj_adaptive_batch[idx] = A_adjusted

        # 自适应邻接矩阵的特定和公共卷积
        adaptive_SGCN_out = self.adaptive_SGCN(x, adj_adaptive_batch)
        adaptive_CGCN_out = self.adaptive_CGCN(x, adj_adaptive_batch)

        # 空间距离邻接矩阵的特定和公共卷积
        spatial_SGCN_out = self.spatial_SGCN(x, adj_spatial)
        spatial_CGCN_out = self.spatial_CGCN(x, adj_spatial)

        # 结合特定卷积和公共卷积的输出
        combined_adaptive = adaptive_SGCN_out + adaptive_CGCN_out
        combined_spatial = spatial_SGCN_out + spatial_CGCN_out

        # 使用注意力机制融合自适应和空间信息
        combined = torch.cat((combined_adaptive, combined_spatial), dim=1)
        attention_out, _ = self.attention(combined)

        # 分类预测
        output = self.classifier(attention_out)
        return F.log_softmax(output, dim=1)




class GraphOperations(nn.Module):
    def __init__(self, num_node_features, num_classes, A_spatial, dropout=0.5):
        super(GraphOperations, self).__init__()
        # 初始化自定义的图卷积网络层
        self.A_spatial = A_spatial
        self.dropout = dropout  # 保存 dropout 值为类的属性
        self.conv1_adaptive = GCN(num_node_features, 512, 256, dropout)
        self.conv2_adaptive = GCN(256, 256, 256, dropout)   #这里的out图卷积输出维度
        self.conv1_spatial = GCN(num_node_features, 512, 256, dropout)
        self.conv2_spatial = GCN(256, 256, 256, dropout)
        self.lin1 = nn.Linear(512, 256)  # 两个图卷积的结果合并

        self.lin = nn.Linear(256, num_classes)  # 两个图卷积的结果合并
        self.MLP = nn.Sequential(
            nn.Linear(128, num_classes),
        )


    def forward(self, batch_features, A_spatial):
        batch_size, num_nodes, num_features = batch_features.size()

        # 动态生成自适应邻接矩阵和处理A_spatial
        adj_adaptive_batch = torch.zeros(batch_size, num_nodes, num_nodes, device=batch_features.device)
        A_spatial_batch = A_spatial.unsqueeze(0).repeat(batch_size, 1, 1)

        x_adaptive_list = []
        x_spatial_list = []

        for idx in range(batch_size):
            features_single = batch_features[idx].unsqueeze(0)  # 加一维变成[1, num_nodes, num_features]

            # 动态生成自适应邻接矩阵
            optimized_A = optimize_adjacency(features_single.squeeze(0))  # 移除batch维度进行处理
            correlation_matrix = compute_pearson_correlation(features_single.squeeze(0))
            A_adjusted = adjust_adjacency_matrix(optimized_A, correlation_matrix)
            adj_adaptive_batch[idx] = A_adjusted
            features_single = features_single.squeeze(0)

            # 打印数据格式
            # print('features_single shape:', features_single.shape)  # 应显示为 [1, num_nodes, num_features]
            # print('A_adjusted.unsqueeze(0) shape:', A_adjusted.shape)  # 应显示为 [1, num_nodes, num_nodes]
            # 自适应图卷积操作
            x_single_adaptive = self.conv1_adaptive(features_single, A_adjusted)
            x_single_adaptive = F.relu(x_single_adaptive)
            x_single_adaptive = F.dropout(x_single_adaptive, self.dropout, training=self.training)
            x_single_adaptive = self.conv2_adaptive(x_single_adaptive, A_adjusted)

            # 空间图卷积操作
            x_single_spatial = self.conv1_spatial(features_single, A_spatial_batch[idx])
            x_single_spatial = F.relu(x_single_spatial)
            x_single_spatial = F.dropout(x_single_spatial, self.dropout, training=self.training)
            x_single_spatial = self.conv2_spatial(x_single_spatial, A_spatial_batch[idx])

            # 收集处理后的结果
            x_adaptive_list.append(x_single_adaptive.squeeze(0))  # 移除额外的维度
            x_spatial_list.append(x_single_spatial.squeeze(0))

        # 将结果从列表转换为张量
        x_adaptive = torch.stack(x_adaptive_list)
        # print('x_adaptive ',x_adaptive.shape)

        x_spatial = torch.stack(x_spatial_list)
        # print('x_spatial ',x_spatial.shape)

        # ##########
        # x_adaptive_agg = torch.mean(x_adaptive, dim=1)  # 形状为[16, 256]
        # x_spatial_agg = torch.mean(x_spatial, dim=1)  # 形状为[16, 256]
        #
        # x_combined = x_adaptive_agg + x_spatial_agg
        # x_combined = self.MLP(x_combined)  # x_class 的形状为[16, 类别数]
        # ###########

        # 融合两种图卷积网络的结果
        x_combined = torch.cat((x_adaptive, x_spatial), dim=-1)
        # print('x_combined ',x_combined.shape)
        # 使用平均池化聚合图级特征
        x_graph_level = torch.mean(x_combined, dim=1)  # 对节点维度进行平均池化
        # print('X%%%%%%',x_graph_level.shape)

        # 将图级特征通过全连接层得到最终的类别预测
        x_out = self.lin1(x_graph_level)  # x_out 的形状将是 [16, 2]

        x_out = self.lin(x_out)  # x_out 的形状将是 [16, 2]

        print('xxx',x_out)
        # x_combined = self.lin(x_combined)
        # print('x_combined ',x_out.shape)


        # return F.log_softmax(x_out, dim=-1)
        return  x_out







