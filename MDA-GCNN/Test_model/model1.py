import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# 定义辅助函数
def optimize_adjacency(features, alpha=0.01, learning_rate=0.01, iterations=100):
    N, D = features.shape
    A = np.random.rand(N, N)
    A = (A + A.T) / 2  # 确保对称性

    for _ in range(iterations):
        X_prime = np.dot(A, features)
        grad_reconstruction = 2 * np.dot(np.dot(A, features) - features, features.T)
        grad_regularization = alpha * np.sign(A)
        A -= learning_rate * (grad_reconstruction + grad_regularization)
        A = np.clip(A, 0, None)
        A = (A + A.T) / 2

    return A

def compute_pearson_correlation(X):
    correlation_matrix = np.corrcoef(X)
    return correlation_matrix


def adjust_adjacency_matrix(A, correlation_matrix, threshold=0.5):
    A_adjusted = np.multiply(A, correlation_matrix)
    A_adjusted[A_adjusted < threshold] = 0
    return A_adjusted


# 定义图卷积网络模型
class EEGGraphConvNet(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(EEGGraphConvNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# 处理单个数据片段的函数
def process_segment(features_np):
    optimized_A = optimize_adjacency(features_np)
    correlation_matrix = compute_pearson_correlation(features_np)
    A_adjusted = adjust_adjacency_matrix(optimized_A, correlation_matrix)

    edge_index = torch.tensor(np.array(np.nonzero(A_adjusted)), dtype=torch.long)
    x = torch.tensor(features_np, dtype=torch.float)

    return Data(x=x, edge_index=edge_index)


# 假设辅助函数和EEGGraphConvNet类定义保持不变

# 修改process_segment函数以处理批次数据
def process_batch(batch_features_np):
    print(f"原始批次数据形状: {batch_features_np.shape}")  # 打印原始数据形状

    batch_data_list = []
    for features_np in batch_features_np:
        optimized_A = optimize_adjacency(features_np)  # 假设函数返回随机邻接矩阵
        correlation_matrix = compute_pearson_correlation(features_np)
        A_adjusted = adjust_adjacency_matrix(optimized_A, correlation_matrix)

        edge_index = torch.tensor(np.array(np.nonzero(A_adjusted)), dtype=torch.long)
        x = torch.tensor(features_np, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        batch_data_list.append(data)

    # 使用Batch来处理图数据的批次
    batch_data = Batch.from_data_list(batch_data_list)
    print(f"节点特征批次形状: {batch_data.x.shape}")  # 打印节点特征批次形状
    print(f"边索引批次形状: {batch_data.edge_index.shape}")  # 打印边索引批次形状
    return batch_data


# 模拟批次数据
batch_size = 32  # 批次大小
num_channels = 62  # 通道数
num_features = 5  # 特征数
simulated_batch_features_np = np.random.rand(batch_size, num_channels, num_features)

# 处理批次数据
batch_data = process_batch(simulated_batch_features_np)

# 初始化模型并进行前向传播
num_node_features = num_features
num_classes = 10  # 假设的类别数
model = EEGGraphConvNet(num_node_features, num_classes)

# 前向传播（假设仅为了形状检查，不考虑实际输出）
out = model(batch_data)
print(f"模型输出形状: {out.shape}")  # 打印模型输出形状

# 假设 `batch_features` 是一个批次的数据，形状为 (batch_size, num_channels, num_features)
# 这里需要根据实际情况加载您的数据
# 示例: batch_features = torch.randn(batch_size, num_channels, num_features)

# 将批次数据转换为图数据列表
# graph_data_list = [process_segment(features.numpy()) for features in batch_features]

# 然后可以创建 DataLoader 来批处理图数据
# data_loader = DataLoader(graph_data_list, batch_size=您的批次大小)

# 注意: 这个示例代码没有包含完整的模型训练循环和数据加载逻辑。
# 您需要根据您的具体任务调整模型的输入、输出和训练过程。
