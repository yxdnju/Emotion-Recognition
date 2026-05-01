import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from Mutiel_view.model_layer1 import *
# from Mutiel_view.model_layer1_1 import *
# from Mutiel_view.model_layer1_2 import *
# from  Mutiel_view.model_layer import  *
# from  Mutiel_view.model_layer1_3 import  *

from DEAP_model.model_layer1 import *


# 基于空间和自适应两种视图的
# class EEGGraphConvNet(nn.Module):
#     def __init__(self, num_node_features, num_classes,A_spatial):
#         super(EEGGraphConvNet, self).__init__()
#         self.A_spatial = A_spatial
#         self.graph_ops = GraphOperations(num_node_features, num_classes,A_spatial)
#
#     def forward(self, batch_features_np):
#         batch_data = self.graph_ops.forward(batch_features_np,self.A_spatial)
#         # print('batch_data',batch_data.shape)
#         return batch_data


class EEGGraphConvNet(nn.Module):
    def __init__(self, num_node_features, num_classes, nhid1, nhid2, dropout, A_spatial):
        super(EEGGraphConvNet, self).__init__()
        # 保存A_spatial为类的属性
        self.A_spatial = A_spatial
        # 初始化GraphOperations实例，传递所有必要的参数
        self.graph_ops = GraphOperations(num_node_features, num_classes, nhid1, nhid2, dropout, A_spatial)

    def forward(self, batch_features_np):
        # 调用GraphOperations实例的forward方法，传入批特征和A_spatial
        batch_data = self.graph_ops.forward(batch_features_np, self.A_spatial)
        # 返回处理后的批数据
        return batch_data
