import torch.nn.functional as F
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from DEAP_model.GCN_layer import *
# 两种视图下的图卷积模型，包括自适应邻接矩阵、空间距离矩阵，
# 在此基础上增加 CNN时间卷积层


def optimize_adjacency(features, alpha=0.3, learning_rate=0.0005, iterations=100):
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






#支持批处理图
    # def forward(self, batch_features, A_spatial):
    #     batch_size, num_nodes, num_features = batch_features.size()
    #
    #     # 动态生成自适应邻接矩阵
    #     adj_adaptive_batch = torch.zeros(batch_size, num_nodes, num_nodes, device=batch_features.device)
    #     for idx in range(batch_size):
    #         features_np = batch_features[idx]
    #         optimized_A = optimize_adjacency(features_np)
    #         correlation_matrix = compute_pearson_correlation(features_np)
    #         A_adjusted = adjust_adjacency_matrix(optimized_A, correlation_matrix)
    #         print('A_adjusted',A_adjusted.shape)
    #         adj_adaptive_batch[idx] = A_adjusted
    #         # 打印输入到图卷积之前的数据格式
    #     # print('batch_features shape:', batch_features.shape)  # 应为 (batch_size, num_nodes, num_features)
    #     # print('adj_adaptive_batch shape:', adj_adaptive_batch.shape)  # 应为 (batch_size, num_nodes, num_nodes)
    #     # print('A_spatial shape before repeat:', A_spatial.shape)  # 应为 (num_nodes, num_nodes)
    #
    #
    #     # 执行自适应图卷积
    #     x_adaptive = self.conv1_adaptive(batch_features, adj_adaptive_batch)
    #     x_adaptive = F.relu(x_adaptive)
    #     x_adaptive = F.dropout(x_adaptive, self.dropout, training=self.training)
    #     x_adaptive = self.conv2_adaptive(x_adaptive, adj_adaptive_batch)
    #
    #     # 执行空间图卷积（注意A_spatial的扩展以匹配批量大小）
    #     A_spatial_batch = A_spatial.unsqueeze(0).repeat(batch_size, 1, 1)
    #     x_spatial = self.conv1_spatial(batch_features, A_spatial_batch)
    #     x_spatial = F.relu(x_spatial)
    #     x_spatial = F.dropout(x_spatial, self.dropout, training=self.training)
    #     x_spatial = self.conv2_spatial(x_spatial, A_spatial_batch)
    #
    #     # 融合两种图卷积网络的结果
    #     x_combined = torch.cat((x_adaptive, x_spatial), dim=-1)
    #     x_combined = self.lin(x_combined)
    #
    #     return F.log_softmax(x_combined, dim=-1)





