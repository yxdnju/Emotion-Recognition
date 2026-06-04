import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
import math
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_scatter import scatter_mean, scatter

class HierarchicalGATConv(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, edge_dim=1,
                 feature_attn_dim=32, **kwargs):
        # 先初始化父类参数
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            edge_dim=edge_dim,  # 必须显式传递
            **kwargs
        )
        
        # 特征级注意力模块,
        # 通过 feature_attn 模块动态学习每个特征维度的权重，放大关键信号（如心电图中的P波特征），抑制噪声
        self.feature_attn = nn.Sequential(
            nn.Linear(in_channels, feature_attn_dim),
            nn.Tanh(),
            nn.Linear(feature_attn_dim, 1, bias=False),
            nn.Sigmoid()
        )
        self.feature_dropout = nn.Dropout(p=0.2)
        # 边权重处理层
        self.edge_weights_lin = nn.Linear(edge_dim, heads)
        
        # 独立参数初始化（避免覆盖父类参数）
        self._init_attn_parameters()

    def _init_attn_parameters(self):
        """专用初始化方法确保参数正确加载"""
        nn.init.xavier_uniform_(self.feature_attn[0].weight)
        nn.init.xavier_uniform_(self.feature_attn[2].weight)
        nn.init.xavier_uniform_(self.edge_weights_lin.weight)

    def reset_parameters(self):
        super().reset_parameters()  # Reset parent's parameters
        # Only initialize custom parameters if they exist
        if hasattr(self, "feature_attn") and hasattr(self, "edge_weights_lin"):
            self._init_custom_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            # Create a default edge attribute tensor. 
            # The shape should be (num_edges, edge_dim) where edge_dim is usually 1.
            edge_attr = torch.ones((edge_index.size(1), 1), device=x.device)
       
        # print("x shape:",x.shape) #x 320*14
        x = x * self.feature_attn(x) #320*1
        x = self.feature_dropout(x)
        return super().forward(x, edge_index, edge_attr=edge_attr)
    
   

def message(self, edge_index_i, x_i, x_j, size_i, edge_attr=None):
    # 如果 edge_attr 为空，可以创建默认值或直接报错
    if edge_attr is None:
        edge_attr = torch.ones((x_i.size(0), 1), device=x_i.device)
    # 原始GAT计算
    x_j = x_j.view(-1, self.heads, self.out_channels)
    alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
    
    # 添加边权重偏置
    if edge_attr is not None:
        edge_weights = self.edge_weights_lin(edge_attr)
        alpha += edge_weights
        
    return alpha.unsqueeze(-1) * x_j
            
    
class AdaptiveGraphPool(nn.Module):
    def __init__(self, in_dim, edge_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim + edge_dim, 1),
            nn.LeakyReLU()
        )
    def forward(self, x, edge_index, edge_attr, batch):
        # 聚合边特征到节点
        row, col = edge_index
        edge_agg = scatter_mean(edge_attr, col, dim=0, dim_size=x.size(0))  # 使用PyG的scatter
        # 联合节点和边特征
        combined = torch.cat([x, edge_agg], dim=1)
        weights = torch.softmax(self.attn(combined), dim=0)
        return global_add_pool(x * weights, batch)
    

class FeatureDisentangler(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.common_fc = nn.Linear(in_dim, in_dim//2)
        
        # 动态权重生成器
        self.dynamic_weight = nn.Sequential(
            nn.Linear(in_dim//2, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )
        
        # 私有特征生成器（增强为两层MLP）
        self.private_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim//2, in_dim//4),
                nn.ReLU(),
                nn.Linear(in_dim//4, in_dim//4)
            ) for _ in range(3)
        ])
        
        self.out_dim = in_dim//2 + 3*(in_dim//4)

    def forward(self, x, labels=None):
        common = F.gelu(self.common_fc(x))
        
        # 动态生成权重
        weights = self.dynamic_weight(common)
        
        # 生成各私有特征
        privates = []
        for i in range(3):
            private = self.private_generators[i](common)
            privates.append(private * weights[:, i].unsqueeze(1))
        
        # 返回私有特征（用于对比损失）
        return torch.cat([common] + privates, dim=1), privates
    
    
class DynamicGNN(nn.Module):
    """集成层次化注意力的动态图神经网络"""
    def __init__(self, input_dim, hidden_dim, output_dim, device,
                 feature_attn_dim=64, gat_heads=4):
        super(DynamicGNN, self).__init__()
        
        # 层次化GAT层
        self.conv1 = HierarchicalGATConv(
            input_dim, hidden_dim, 
            heads=gat_heads,
            edge_dim=1,
            feature_attn_dim=feature_attn_dim,
           
        )
        # self.conv1 = GATConv(
        #     input_dim, hidden_dim, 
        #     heads=gat_heads,
        #     edge_dim=1,
           
           
        # )
        # SAGE卷积层,平衡局部与全局特征,对边噪声不敏感
        self.conv2 = SAGEConv(hidden_dim * gat_heads, output_dim, aggr='mean')
        
         # 归一化层
        self.bn1 = nn.BatchNorm1d(hidden_dim * gat_heads)
        self.bn2 = nn.BatchNorm1d(output_dim)

        # 节点重要性预测器
        self.node_weight_predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
         # 增加残差连接,将原始特征的低阶信息（如信号原始波形）直接传递到深层，缓解梯度消失
        self.res_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * gat_heads),
            nn.LayerNorm(hidden_dim * gat_heads),
            nn.ReLU()
        )

        # 增加跨层门控机制,自动学习原始特征与GAT输出的融合比例
        self.gate = nn.Linear(2 * hidden_dim * gat_heads, 1)
        self.pool = AdaptiveGraphPool(output_dim,edge_dim=1)
        # 特征解耦器
        self.disentangler = FeatureDisentangler(output_dim)
        classifier_input_dim = self.disentangler.out_dim
        # 特征提取与分类器
        # self.feature_extraction = self._build_feature_extractor(output_dim, hidden_dim)
        self.classifier = self._build_classifier(classifier_input_dim, hidden_dim)
        
        # 初始化参数
        self.apply(self._init_weights)
        self.device = device

        # 消融
        self.fc = nn.Linear(input_dim, output_dim)
        self.rc = nn.Linear(128,160)
    # def _build_feature_extractor(self, input_dim, hidden_dim):
    #     return nn.Sequential(
    #         nn.Linear(input_dim, hidden_dim*4),
    #         nn.BatchNorm1d(hidden_dim*4),
    #         nn.GELU(),
    #         nn.Dropout(0.5),
    #         nn.Linear(hidden_dim*4, 3)
    #     )

    def _build_classifier(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 第一层映射到hidden_dim
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 128)          # 最终映射到目标维度
        )


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, edge_index, edge_attr, batch,y):
        
        # 边特征处理
        edge_attr = edge_attr.view(-1, 1).to(self.device)
        x =  torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6) 
        # 残差连接
        residual = self.res_fc(x)
        # 第一层：层次化GAT
        h1 = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr))) 
        # 门控融合
        gate = torch.sigmoid(self.gate(torch.cat([h1, residual], dim=1)))
        h1 = gate * h1 + (1 - gate) * residual
        h1 = F.dropout(h1, p=0.3, training=self.training)
        
        # 第二层：SAGE卷积
        h2 = F.elu(self.bn2(self.conv2(h1, edge_index)))
        h2 = F.dropout(h2, p=0.3, training=self.training)

        # # 节点重要性加权
        # node_weights = self.node_weight_predictor(h2)
        # weighted_features = h2 * node_weights
        # 消融实验
        # h2 = self.fc(x)
        # 图级别聚合
        graph_embeddings = self.pool(h2, edge_index, edge_attr, batch)
        
        # 特征解耦（返回解耦特征和私有特征）
        disentangled, private_features = self.disentangler(graph_embeddings, labels=y)
        
        return self.classifier(disentangled), private_features  # 返回私有特征用于损失


    def forward_with_intermediate(self, x, edge_index, edge_attr, batch, y):
        """返回中间特征用于可视化分析"""
        # 边特征处理
        edge_attr = edge_attr.view(-1, 1).to(self.device)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 残差连接
        residual = self.res_fc(x)
        
        # 第一层：层次化GAT
        h1 = F.elu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        gate = torch.sigmoid(self.gate(torch.cat([h1, residual], dim=1)))
        h1 = gate * h1 + (1 - gate) * residual
        h1 = F.dropout(h1, p=0.3, training=self.training)
        
        # 第二层：SAGE卷积
        h2 = F.elu(self.bn2(self.conv2(h1, edge_index)))
        h2 = F.dropout(h2, p=0.3, training=self.training)
        
        # 图级别聚合
        graph_embeddings = self.pool(h2, edge_index, edge_attr, batch)
        
        # 获取特征级注意力权重（用于注意力演变可视化）
        feature_attn_weights = self.conv1.feature_attn(x).detach()
        
        # 特征解耦
        common = self.disentangler.common_fc(graph_embeddings)
        
        # 获取私有特征
        private_features = []
        for i in range(3):
            private = self.disentangler.private_generators[i](common)
            private_features.append(private)
        
        # 动态权重
        weights = self.disentangler.dynamic_weight(common)
        
        # 加权私有特征
        weighted_privates = []
        for i in range(3):
            weighted_privates.append(private_features[i] * weights[:, i].unsqueeze(1))
        
        # 解耦后的特征
        disentangled = torch.cat([common] + weighted_privates, dim=1)
        logits = self.classifier(disentangled)
        
        return {
            'logits': logits,
            'common_features': common,
            'private_features': private_features,      # 未加权的私有特征
            'weighted_privates': weighted_privates,    # 加权的私有特征
            'dynamic_weights': weights,
            'graph_embeddings': graph_embeddings,
            'feature_attn_weights': feature_attn_weights
        }
   

    # 辅助方法
    def get_feature_importance(self):
        """获取特征级注意力权重"""
        return self.conv1.feature_attn[2].weight.detach().cpu().numpy()

    def get_node_importance(self):
        """获取节点重要性权重"""
        return self.node_weight_predictor[-2].weight.detach().cpu().numpy()

class TemporalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, time_dim=64):
        super().__init__()
        # 基础GNN模块
        self.gnn = DynamicGNN(input_dim, hidden_dim, output_dim, device)
      
        # 时间编码器
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, time_dim)
        )
        
        # 跨图注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=192 ,
            num_heads=4
        )
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),                  # 降低Dropout比例
            nn.Linear(128, 64),               # 逐步降维
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 3) 
        )
        self.fc_time = nn.Linear(64, 128)
    def forward(self, batch):
        # 基础GNN特征（现在返回两个值）
        base_feat, private_features = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.y)
        
        # 时间特征
        time_diff = self._calc_time_diff(batch)
        time_emb = self.time_encoder(time_diff.unsqueeze(-1))
        
        # 确保 time_emb 是 2D
        if time_emb.dim() == 3:
            time_emb = time_emb.squeeze(1)  # 移除序列维度
        # 特征融合
        fused_feat = torch.cat([base_feat, time_emb], dim=1)
        
        # 跨图注意力
        attn_out, _ = self.cross_attn(
            fused_feat.unsqueeze(0),
            fused_feat.unsqueeze(0),
            fused_feat.unsqueeze(0)
        )
        attn_out = F.dropout(attn_out.squeeze(0), p=0.4, training=self.training)
        logits = self.classifier(attn_out)
        
        return logits, private_features  # 返回私有特征
        
    def forward_with_intermediate(self, batch):
        """返回中间特征用于可视化分析"""
        # 基础GNN特征
        gnn_output = self.gnn.forward_with_intermediate(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.y
        )
        
        # 时间特征
        time_diff = self._calc_time_diff(batch)
        time_emb = self.time_encoder(time_diff.unsqueeze(-1))
        
        # 特征融合
        base_feat = gnn_output['graph_embeddings']
        fused_feat = torch.cat([base_feat, time_emb], dim=1)
        
        # 跨图注意力
        attn_out, _ = self.cross_attn(
            fused_feat.unsqueeze(0),
            fused_feat.unsqueeze(0),
            fused_feat.unsqueeze(0)
        )
        attn_out = F.dropout(attn_out.squeeze(0), p=0.4, training=self.training)
        
        # 分类
        logits = self.classifier(attn_out)
        
        return {
            'logits': logits,
            'common_features': gnn_output['common_features'],
            'private_features': gnn_output['private_features'],
            'weighted_privates': gnn_output['weighted_privates'],
            'dynamic_weights': gnn_output['dynamic_weights'],
            'feature_attn_weights': gnn_output['feature_attn_weights'],
            'time_embeddings': time_emb
        }
    def _calc_time_diff(self, batch):
        """
        按受试者独立计算相对时间差
        即使 batch 中的样本是打乱的，也能正确按受试者分组
        
        Args:
            batch: PyG Batch 对象，包含 subject_id 和 timestamp 属性
            
        Returns:
            time_diff: [batch_size, 1] 相对时间差
        """
        # 1. 检查是否有受试者ID
        if not hasattr(batch, 'subject_id') or batch.subject_id is None:
            return torch.zeros((batch.num_graphs, 1), device=batch.x.device)
        
        # 2. 确保 subject_id 是一维的
        subject_ids = batch.subject_id
        if subject_ids.dim() > 1:
            subject_ids = subject_ids.squeeze()
        
        # 3. 获取所有唯一的受试者ID
        unique_subjects = torch.unique(subject_ids)
        
        # 4. 初始化输出
        time_diff = torch.zeros((batch.num_graphs, 1), device=batch.x.device)
        
        # 5. 遍历每个受试者
        for subj in unique_subjects:
            # 找到属于当前受试者的样本索引
            mask = (subject_ids == subj)
            
            # 获取这些样本的时间戳
            subj_timestamps = batch.timestamp[mask]
            
            if len(subj_timestamps) > 0:
                # 计算相对时间（减去该受试者的最小时间戳）
                subj_time_diff = subj_timestamps - subj_timestamps.min()
                time_diff[mask] = subj_time_diff
        
        return time_diff

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()

# 在原始模型中使用位置编码
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalPositionalEncoding, self).__init__()
        self.pos_encoder = PositionalEncoding(hidden_dim)
    
    def forward(self, x):
        return self.pos_encoder(x)
