import numpy as np
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class EEGGraphConvNet(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(EEGGraphConvNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, batch_features):
        batch_data_list = []

        # 处理每个批次中的数据
        for features in batch_features:
            # 转换为NumPy数组以用于自适应邻接矩阵生成
            features_np = features.numpy()

            # 优化邻接矩阵
            optimized_A = optimize_adjacency(features_np)
            # 计算皮尔逊相关系数矩阵
            correlation_matrix = compute_pearson_correlation(features_np)
            # 调整邻接矩阵
            A_adjusted = adjust_adjacency_matrix(optimized_A, correlation_matrix)

            # 转换回PyTorch张量
            edge_index = torch.tensor(np.array(A_adjusted.nonzero()), dtype=torch.long)
            x = torch.tensor(features_np, dtype=torch.float)

            # 创建图数据对象
            data = Data(x=x, edge_index=edge_index)
            batch_data_list.append(data)

        # 假设所有数据片段具有相同的图结构
        # 使用第一个片段的邻接矩阵作为整个批次的邻接矩阵
        edge_index = batch_data_list[0].edge_index
        x = torch.cat([data.x for data in batch_data_list], dim=0)

        # 图卷积网络前向传播
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# 注意：这个模型示例假设所有数据片段具有相同的图结构，
# 在实际应用中，可能需要为每个片段单独处理图结构。



def optimize_adjacency(features, alpha=0.01, learning_rate=0.01, iterations=100):
    N, D = features.shape
    # 初始化邻接矩阵A为随机值，并保证其对称性
    A = np.random.rand(N, N)
    A = (A + A.T) / 2

    for iteration in range(iterations):
        # 计算图传播后的特征X'，这里简化为A和特征矩阵的乘积
        X_prime = np.dot(A, features)

        # 计算重构误差的梯度
        grad_reconstruction = 2 * np.dot(np.dot(A, features) - features, features.T)

        # 计算L1正则化的梯度
        grad_regularization = alpha * np.sign(A)

        # 根据梯度更新A
        A -= learning_rate * (grad_reconstruction + grad_regularization)

        # 保证A的权重非负
        A = np.clip(A, 0, None)

        # 保持A的对称性
        A = (A + A.T) / 2

    return A

def compute_pearson_correlation(X):
    """计算皮尔逊相关系数矩阵"""
    correlation_matrix = np.corrcoef(X)
    return correlation_matrix


def adjust_adjacency_matrix(A, correlation_matrix, threshold=0.5):
    """
    根据皮尔逊相关系数调整邻接矩阵A。
    A: 初始邻接矩阵
    correlation_matrix: 通道间的皮尔逊相关系数矩阵
    threshold: 阈值，用于过滤低相关性的连接
    """
    # 将相关性强度作为权重
    A_adjusted = np.multiply(A, correlation_matrix)

    # 应用阈值，过滤微弱连接
    A_adjusted[A_adjusted < threshold] = 0

    return A_adjusted


# 示例数据X，假设已经加载
# 计算相关性矩阵
correlation_matrix = compute_pearson_correlation(X)

# 调整邻接矩阵
A_adjusted = adjust_adjacency_matrix(A, correlation_matrix, threshold=0.5)

# 假设features是节点特征矩阵，形状为(N, D)，其中N是节点数，D是特征维度
# 例如：
# features = np.random.rand(5, 3) # 5个节点，每个节点3个特征

# alpha = 0.01  # 正则化系数，控制稀疏性强度
# learning_rate = 0.01  # 学习率，控制每次迭代中A更新的步长
# iterations = 100  # 最大迭代次数

# 调用函数优化邻接矩阵
# optimized_A = optimize_adjacency(features, alpha, learning_rate, iterations)
