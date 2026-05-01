import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv,global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def optimize_adjacency(features, alpha=0.3, learning_rate=0.001, iterations=100):
    device = features.device  # 获取输入张量的设备信息
    N, D = features.shape
    A = torch.rand(N, N, device=device)  # 使用torch.rand创建张量，并将其移动到正确的设备上
    A = (A + A.t()) / 2  # 确保对称性

    for _ in range(iterations):
        X_prime = torch.matmul(A, features)  # 使用torch.matmul代替np.dot
        grad_reconstruction = 2 * torch.matmul(torch.matmul(A, features) - features, features.t())
        grad_regularization = alpha * torch.sign(A)
        A -= learning_rate * (grad_reconstruction + grad_regularization)
        A = A.clamp(0)  # 使用torch.clamp代替np.clip
        A = (A + A.t()) / 2

    return A


def compute_pearson_correlation(X):
    # 将 NumPy 数组转换为 PyTorch 张量
    X_tensor = X.clone().detach()
    # X_tensor = torch.tensor(X)
    # 在 GPU 上计算相关系数矩阵
    correlation_matrix = torch.corrcoef(X_tensor)
    # 将相关系数矩阵转换回 NumPy 数组
    # correlation_matrix_numpy = correlation_matrix.cpu().numpy()
    return correlation_matrix

def adjust_adjacency_matrix(A, correlation_matrix, threshold=0.5):
    # 将 NumPy 数组转换为 PyTorch 张量
    # A_tensor = torch.tensor(A)
    A_tensor = A.clone().detach()
    # correlation_matrix_tensor = torch.tensor(correlation_matrix)
    correlation_matrix_tensor = correlation_matrix.clone().detach()
    # 在 GPU 上执行相关操作
    A_adjusted = torch.mul(A_tensor, correlation_matrix_tensor)
    A_adjusted[A_adjusted < threshold] = 0

    # 将结果转回 NumPy 数组
    return A_adjusted  # 如果需要将结果返回到 CPU 上，可以使用 .cpu() 方法
#
# class GraphOperations(torch.nn.Module):
#     def __init__(self, num_node_features, num_classes):
#         super(GraphOperations, self).__init__()
#         self.conv1 = GCNConv(num_node_features, 16)
#         self.conv2 = GCNConv(16, num_classes)
#
#     def forward(self, batch_features_np):
#         # batch_features_np = batch_features_np[0]
#         # print('batch========',batch_features_np.shape)
#         # print("Data type:", type(batch_features_np))
#         # if isinstance(batch_features_np, list):
#         #     # 假设列表中所有元素都有相同的数据格式
#         #     if len(batch_features_np) > 0:
#         #         print("Element type:", type(batch_features_np[0]))
#         #         if isinstance(batch_features_np[0], np.ndarray):
#         #             print("Element shape:", batch_features_np[0].shape)
#         #         elif isinstance(batch_features_np[0], torch.Tensor):
#         #             print("Element shape:", batch_features_np[0].shape)
#         #         else:
#         #             print("Element is neither a numpy array nor a tensor.")
#         #     else:
#         #         print("List is empty.")
#         # else:
#         #     print("Data is not a list.")
#
#         batch_data = self.process_batch(batch_features_np)
#         x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         # 使用全局平均池化将节点级别的预测聚合成图级别的预测
#         x = torch_geometric.nn.global_mean_pool(x, batch)
#         return F.log_softmax(x, dim=1)

class GraphOperations(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GraphOperations, self).__init__()
        self.conv1 = GCNConv(num_node_features, 256)  # 第一层图卷积，输出特征维度为16
        self.conv2 = GCNConv(256, 128)  # 第二层图卷积，输出特征维度为32
        self.lin = torch.nn.Linear(128, num_classes)  # 线性变换层，从32维特征到类别数的映射

    def forward(self, batch_features_np):
        batch_data = self.process_batch(batch_features_np)
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
        x = F.relu(self.conv1(x, edge_index))  # 第一层图卷积后应用ReLU激活函数
        x = F.dropout(x, p=0.5, training=self.training)  # 应用dropout
        x = F.relu(self.conv2(x, edge_index))  # 第二层图卷积后应用ReLU激活函数
        x = pyg_nn.global_mean_pool(x, batch)  # 注意这一行的变化

        x = self.lin(x)  # 应用线性变换层
        # print("GCN ",x.shape)
        return F.log_softmax(x, dim=1)  # 返回log-softmax作为最终的输出

    def process_batch(self, batch_features_np):
        batch_data_list = []
        for features_np in  batch_features_np:
        # features_np = batch_features_np
            # 如果features_np是numpy数组，确保在此转换为张量

            # print('feature_np',features_np.shape)
            # print("feature_np Element type:", type(features_np))

            optimized_A = optimize_adjacency(features_np)
            correlation_matrix = compute_pearson_correlation(features_np)
            A_adjusted = adjust_adjacency_matrix(optimized_A, correlation_matrix)

            # if torch.nonzero(A_adjusted).size(0) > 0:
            # A_adjusted 中存在非零元素
            #     print("A_adjusted 中存在非零元素")
            # else:
            # A_adjusted 中所有元素都为零
            #     print("A_adjusted 中所有元素都为零")


            # edge_index = torch.tensor(np.array(np.nonzero(A_adjusted)), dtype=torch.long)
        # Assuming A_adjusted is a PyTorch tensor
            edge_index = torch.nonzero(A_adjusted, as_tuple=False).t().contiguous()
            edge_weights = A_adjusted[edge_index[0],edge_index[1]]

        #     edge_index = A_adjusted.nonzero(as_tuple=)
        # 转换 edge_weights 为正确的形状 (num_edges, num_features)
        # 这里我们假设每个边的权重就是其特征，如果有多个特征需要不同的处理
            edge_weights = edge_weights.view(-1, 1)  # 假设每条边只有一个特征
            # print('A_adjust', A_adjusted.shape)
            # print('edge_weights', edge_weights.shape)
            # print('edge_index', edge_index.shape)


        # x = torch.tensor(features_np, dtype=torch.float)
            x = features_np.clone().detach().float()  # 假设features_np已经是一个张量
            data = Data(x=x, edge_index=edge_index,edge_attr=edge_weights)
            batch_data_list.append(data)
            # print('batch_data_list',batch_data_list[0])

        batch_data = Batch.from_data_list(batch_data_list)

        return batch_data