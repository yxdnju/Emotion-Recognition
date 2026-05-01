import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from Mutiel_view.model_layer1 import *
from Mutiel_view.model_layer1_1 import *
# from Mutiel_view.model_layer1_2 import *
# from  Mutiel_view.model_layer import  *
# from  Mutiel_view.model_layer1_3 import  *


from torch_geometric.data import Data, Batch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from  SEED_Distance import  *


#基于特征
# class EEGGraphConvNet(nn.Module):
#     def __init__(self, num_node_features, num_classes):
#         super(EEGGraphConvNet, self).__init__()
#         self.graph_ops = GraphOperations(num_node_features, num_classes)
#
#     def forward(self, batch_features_np):
#         batch_data = self.graph_ops.forward(batch_features_np)
#         # print('batch_data',batch_data.shape)
#         return batch_data


# 基于空间和自适应两种视图的
class EEGGraphConvNet(nn.Module):
    def __init__(self, num_node_features, num_classes,A_spatial):
        super(EEGGraphConvNet, self).__init__()
        self.A_spatial = A_spatial
        self.graph_ops = GraphOperations(num_node_features, num_classes,A_spatial)

    def forward(self, batch_features_np):
        batch_data = self.graph_ops.forward(batch_features_np,self.A_spatial)
        # print('batch_data',batch_data.shape)
        return batch_data

# # +时间卷积层
# class EEGGraphTemporalConvNet(nn.Module):
#     def __init__(self, num_node_features, num_temporal_features, num_classes, A_spatial):
#         super(EEGGraphTemporalConvNet, self).__init__()
#         self.A_spatial = A_spatial
#
#         self.graph_ops = GraphOperations(num_node_features, 256 , A_spatial)  # 假设图卷积输出特征维度为256+128
#         self.temp_conv_net = TemporalConvNet(256 +128, num_temporal_features, num_classes)
#
#     def forward(self, batch_features_np):
#         # 使用GraphOperations处理空间信息
#         spatial_features = self.graph_ops(batch_features_np, self.A_spatial)
#
#         # 调整维度以符合时间卷积网络的输入要求
#         # 假设你想要的维度是 [批次大小, 通道数, 高度(虚拟的，可以是1), 宽度(时间维度)]
#         spatial_features = spatial_features.unsqueeze(2)  # 添加一个虚拟的维度作为"高度"
#         spatial_features = spatial_features.unsqueeze(3)  # 假设每个时间点只有一个时间步长
#
#         # 然后使用TemporalConvNet处理时间信息
#         output = self.temp_conv_net(spatial_features)
#
#         # print('output',output.shape)
#
#         return output

#  二维时间卷积
# # 整个模型结构
# class EEGGraphTemporalConvNet(nn.Module):
#     def __init__(self, num_node_features, num_temporal_features, num_classes, A_spatial):
#         super(EEGGraphTemporalConvNet, self).__init__()
#         self.A_spatial = A_spatial
#         self.graph_ops = GraphOperations(num_node_features, 256, A_spatial)
#         self.temp_conv_net = TemporalConvNet(num_temporal_features, num_classes)
#
#     def forward(self, batch_features_np):
#         # 图卷积处理
#         spatial_features = self.graph_ops(batch_features_np,self.A_spatial)
#         # 时间卷积处理
#         output = self.temp_conv_net(spatial_features)
#         # print('output',output.shape)
#         return output
#
#
#
# # 加载数据和标签
# x = np.load("G:/Experiment/Experiment3/Data/SEED/x.npy")
# y = np.load("G:/Experiment/Experiment3/Data/SEED/y.npy")
#
# # 将数据转换为torch.Tensor
# x_tensor = torch.tensor(x, dtype=torch.float)
# y_tensor = torch.tensor(y, dtype=torch.long)
#
# # 创建TensorDataset和DataLoader
# dataset = TensorDataset(x_tensor, y_tensor)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# # 获取一个batch的数据
# features, labels = next(iter(loader))
#
# # 创建基于空间距离的邻接矩阵A_spatial
# # 这里需要您根据实际情况计算或加载
# electrodes = Electrodes62()  # 确保已经正确实现了这个类
# A_spatial_numpy = electrodes.get_adjacency_matrix()
# A_spatial = torch.tensor(A_spatial_numpy, dtype=torch.float)
# # 确认邻接矩阵的形状
# print(A_spatial.shape)
#
#
# # 初始化模型并设置为评估模式
# model = EEGGraphTemporalConvNet(num_node_features=5, num_temporal_features=32, num_classes=3, A_spatial=A_spatial)
# model.eval()
#
# # 使用模型进行预测
# with torch.no_grad():
#     output = model(features)
#     print('模型输出',output.shape)  # 应该输出(batch_size, num_classes)的形状，即(32, 3)


#
# # 假设有一个batch大小为4，每个样本有62个通道，每个通道有5个特征
# batch_size = 32
# num_channels = 62
# num_features = 5
#
# # 创建模拟输入数据
# features_np = np.random.rand(batch_size, num_channels, num_features).astype(np.float32)
#
# # 转换为Tensor
# features_tensor = torch.tensor(features_np)
#
# # 创建基于空间距离的邻接矩阵，这里简单模拟一个邻接矩阵
# # 实际应用中，您需要使用您计算得到的空间邻接矩阵
# A_spatial_numpy = np.random.rand(num_channels, num_channels)
# A_spatial = torch.tensor(A_spatial_numpy, dtype=torch.float)
#
# # 初始化模型
# # 请确保您的模型初始化接口与这里提供的相匹配
# model = EEGGraphTemporalConvNet(num_node_features=num_features, num_classes=3, A_spatial=A_spatial)
#
# # 将模型设置为评估模式
# model.eval()
#
# # 假设您的模型的forward方法接受单个batch的特征数据和A_spatial作为输入
# # 这里使用for循环模拟批处理
# for i in range(batch_size):
#     single_batch_features = features_tensor[i].unsqueeze(0)  # 添加batch维度
#     output = model(single_batch_features, A_spatial)
#     print(f"Output for batch {i}: {output.shape}")