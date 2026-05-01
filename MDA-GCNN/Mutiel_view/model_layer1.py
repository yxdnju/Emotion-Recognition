import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 两种视图下的图卷积模型，包括自适应邻接矩阵、空间距离矩阵


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
    return correlation_matrix

def adjust_adjacency_matrix(A, correlation_matrix, threshold=0.5):
    A_tensor = A.clone().detach()
    correlation_matrix_tensor = correlation_matrix.clone().detach()
    A_adjusted = torch.mul(A_tensor, correlation_matrix_tensor)
    A_adjusted[A_adjusted < threshold] = 0
    return A_adjusted

# 假设已经有了Electrodes62类和相关计算距离的函数

class GraphOperations(torch.nn.Module):
    def __init__(self, num_node_features, num_classes,A_spatial):
        super(GraphOperations, self).__init__()
        self.A_spatial = A_spatial
        # 对于自适应邻接矩阵的图卷积层
        self.conv1_adaptive = GCNConv(num_node_features, 256)
        self.conv2_adaptive = GCNConv(256, 128)

        # 对于基于空间距离的邻接矩阵的图卷积层
        self.conv1_spatial = GCNConv(num_node_features, 256)
        self.conv2_spatial = GCNConv(256, 128)

        # 线性层用于融合两个图卷积网络的结果
        self.lin = nn.Linear(256, num_classes)

    def forward(self, batch_features_np, A_spatial):
        # 处理自适应生成的邻接矩阵
        batch_data_adaptive = self.process_batch(batch_features_np)
        x_adaptive, edge_index_adaptive, batch = batch_data_adaptive.x, batch_data_adaptive.edge_index, batch_data_adaptive.batch
        x_adaptive = F.relu(self.conv1_adaptive(x_adaptive, edge_index_adaptive))
        x_adaptive = F.dropout(x_adaptive, p=0.5, training=self.training)
        x_adaptive = F.relu(self.conv2_adaptive(x_adaptive, edge_index_adaptive))
        x_adaptive = global_mean_pool(x_adaptive, batch)


        # 处理基于空间距离的邻接矩阵
        batch_data_spatial = self.process_batch_with_spatial(batch_features_np, A_spatial)
        x_spatial, edge_index_spatial, batch = batch_data_spatial.x, batch_data_spatial.edge_index, batch_data_spatial.batch
        x_spatial = F.relu(self.conv1_spatial(x_spatial, edge_index_spatial))
        x_spatial = F.dropout(x_spatial, p=0.5, training=self.training)
        x_spatial = F.relu(self.conv2_spatial(x_spatial, edge_index_spatial))
        x_spatial = global_mean_pool(x_spatial, batch)

        # 融合两种图卷积网络的结果
        x_combined = torch.cat((x_adaptive, x_spatial), dim=1)
        x_combined = self.lin(x_combined)

        return F.log_softmax(x_combined, dim=1)


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
