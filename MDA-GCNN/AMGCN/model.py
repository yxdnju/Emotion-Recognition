import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = self.fc(x)
        output = torch.bmm(adj, support)
        return output


def build_knn_graph(x, k=10, mode='cosine'):
    # 假设x的形状为 [num_nodes, num_features]
    sim_matrix = torch.mm(x, x.t())
    if mode == 'cosine':
        norms = torch.norm(x, p=2, dim=1)
        sim_matrix = sim_matrix / (norms.unsqueeze(1) * norms.unsqueeze(0))

    # 选取每个节点的top k个邻居
    _, indices = torch.topk(sim_matrix, k=k + 1, dim=-1)
    adjacency = torch.zeros_like(sim_matrix)
    adjacency.scatter_(1, indices, 1)
    return adjacency


class CommonGCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(CommonGCN, self).__init__()
        self.gc1 = GraphConvolution(in_features, out_features)
        self.gc2 = GraphConvolution(out_features, out_features)

    def forward(self, x, adj_topo, adj_feat):
        x_topo = F.relu(self.gc1(x, adj_topo))
        x_feat = F.relu(self.gc1(x, adj_feat))
        x_common_topo = F.relu(self.gc2(x_topo, adj_topo))
        x_common_feat = F.relu(self.gc2(x_feat, adj_feat))
        x_common = (x_common_topo + x_common_feat) / 2
        return x_common

class AttentionLayer(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(AttentionLayer, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.W = nn.Parameter(torch.Tensor(in_features, hidden_features))
        self.q = nn.Parameter(torch.Tensor(hidden_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, ZT, ZC, ZF):
        Z = torch.stack([ZT, ZC, ZF], dim=1)  # [num_nodes, 3, in_features]
        Z_transformed = torch.tanh(torch.matmul(Z, self.W))  # [num_nodes, 3, hidden_features]
        scores = torch.matmul(Z_transformed, self.q).squeeze(-1)  # [num_nodes, 3]
        attention_weights = F.softmax(scores, dim=-1)
        return attention_weights

class AMGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(AMGCN, self).__init__()
        self.specific_gcn = GraphConvolution(nfeat, nhid)
        self.common_gcn = CommonGCN(nfeat, nhid)
        self.attention = AttentionLayer(nhid, nhid // 2)
        self.dropout = dropout
        self.fc = nn.Linear(nhid * 3, nclass)

    def forward(self, x, adj_topo, adj_feat,adj_knn=None):
        # 如果没有提供 adj_knn，就使用特征图的邻接矩阵构建它
        if adj_knn is None:
            adj_knn = build_knn_graph(x, k=10, mode='cosine')

        ZF = F.relu(self.specific_gcn(x, adj_feat))
        ZT = F.relu(self.specific_gcn(x, adj_topo))
        ZC = self.common_gcn(x, adj_topo, adj_feat)
        attention_weights = self.attention(ZT, ZC, ZF)
        Z_final = torch.cat((ZF * attention_weights[:, 0:1],
                             ZC * attention_weights[:, 1:2],
                             ZT * attention_weights[:, 2:3]), dim=1)
        Z_final = F.dropout(Z_final, self.dropout, training=self.training)
        out = self.fc(Z_final)
        return F.log_softmax(out, dim=1)


