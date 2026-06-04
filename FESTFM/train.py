from matplotlib import pyplot as plt
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import CustomLoss
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from event_detector import EventDetector
from collections import Counter
from extract_features_graph_builder import build_graph_data
from functools import partial

class PrivateContrastiveLoss(nn.Module):
    """私有特征对比损失：强制同类靠近、异类远离"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, private_features, labels):
        """
        private_features: list of 3 tensors, each [batch, dim]
        labels: [batch]
        """
        # 拼接所有私有特征
        all_privates = torch.cat(private_features, dim=1)  # [batch, dim*3]
        all_privates = F.normalize(all_privates, p=2, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(all_privates, all_privates.T) / self.temperature
        
        # 构建正样本掩码（相同标签）
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # 计算对比损失
        exp_sim = torch.exp(sim_matrix)
        pos_sim = (exp_sim * mask).sum(dim=1)
        neg_sim = exp_sim.sum(dim=1) - pos_sim
        
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
        return loss.mean()    
def evaluate_dynamic_gnn(model, test_dataset, criterion, fs, device, batch_size=32):
    model.eval()
    total_loss = 0
    total_samples = 0  # 用于累计总样本数
    all_predictions = []
    all_true_labels = []



    # 使用 PyGDataLoader 对测试数据集进行批处理加载
    test_loader = PyGDataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        # collate_fn=partial(fs=fs, device=device)
    )
        # 启用 Dropout 和 BatchNorm 的动态行为（如果需要）
    for module in model.modules():
        # if isinstance(module, nn.Dropout) or isinstance(module, nn.BatchNorm1d):
        #     module.train()
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    # 前向传播
                    output = model(batch)
                    
                    # 处理返回值：如果是元组，取第一个元素作为预测值
                    if isinstance(output, tuple):
                        predictions = output[0]  # 取 logits
                    else:
                        predictions = output
                    
                    # 计算损失
                    loss = criterion(predictions, batch.y)
                    
                    # 累加损失，按样本数加权
                    total_loss += loss.item() * batch.num_graphs
                    total_samples += batch.num_graphs
                    
                    # 获取批次中每个图的预测类别
                    batch_preds = torch.argmax(predictions, dim=1)
                    all_predictions.append(batch_preds)
                    all_true_labels.append(batch.y)

            # 合并所有批次的预测和标签
            all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
            all_true_labels = torch.cat(all_true_labels, dim=0).cpu().numpy()

            # 计算平均损失
            avg_loss = total_loss / total_samples  # 按样本数计算平均损失

            # 计算评估指标
            accuracy = accuracy_score(all_true_labels, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average='weighted')
            cm = confusion_matrix(all_true_labels, all_predictions)

            # 打印和保存评估指标
            print_metrics(accuracy, precision, recall, f1, cm, "Evaluation Metrics")
            save_confusion_matrix(cm, "confusion_matrix.png", "Confusion Matrix")

            return {
                'accuracy': accuracy, 
                'precision': precision, 
                'recall': recall, 
                'f1': f1, 
                'confusion_matrix': cm,
                'val_loss': avg_loss,
                'cm': cm
            }

  
    
# ===== 新增代码 =====
def curriculum_scheduler(optimizer,total_epochs=100):
    """改进后的调度器"""
    def lr_lambda(epoch):
        if epoch < 15:
            return 0.1 + 0.9*(epoch/15)
        elif epoch < 40:
            return 1.0
        else:
            decay_epochs = total_epochs - 40
            progress = min((epoch - 40) / decay_epochs, 1.0)
            return max(0.001, 1.0 - 0.99*progress)  # 更平缓的衰减
    return LambdaLR(optimizer, lr_lambda)

def train_dynamic_gnn_with_events(model, train_dataset,  directory_path, num_classes=3, num_epochs=50, lr=0.1,batch_size=32,fs=4,device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # scheduler = curriculum_scheduler(optimizer,num_epochs)
    # 学习率调度：线性预热 + 余弦退火
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=3e-3,
        epochs=num_epochs,
        steps_per_epoch = len(train_dataset) // batch_size + int(len(train_dataset) % batch_size != 0),

        pct_start=0.05  # 前10%的迭代用于学习率预热
    )
    # # 修改后的优化器配置（分阶段学习率探测）
    # optimizer = torch.optim.AdamW(
    #     [
    #         {'params': model.gnn.conv1.parameters(), 'lr': 5e-5},  # 极低学习率探测卷积层
    #         {'params': model.gnn.disentangler.parameters(), 'lr': 1e-4},
    #         {'params': model.classifier.parameters(), 'lr': 2e-4}
    #     ],
    #     weight_decay=0.01  # 增强权重衰减
    # )

    # # 修改学习率调度为热重启策略
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=5,  # 每5个epoch重启一次
    #     T_mult=2,
    #     eta_min=1e-6  # 最低学习率
    # )
    # class_weights = torch.tensor([1.0, 1.0, 5.0], device=device)  # 第三类权重设为5
    
    criterion = nn.CrossEntropyLoss(ignore_index=4)
    # criterion = CustomLoss.CustomLoss()
    # criterion = nn.CosineEmbeddingLoss(margin=0.5)
    n_windows = len(train_dataset)
    

    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    all_complexities = []
    global_significant_events = []
   

    file_name = "training_log.txt"

    # Combine the directory path and file name
    log_file = f"{directory_path}/{file_name}"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Training Log\n")
        f.write("===========================\n")
    


    
    # 初始化私有特征对比损失
    private_contrastive_loss = PrivateContrastiveLoss(temperature=0.07)
   # 使用自定义的 collate_fn 进行数据加载
    pyg_loader = PyGDataLoader(train_dataset, batch_size=batch_size)

    # 训练阶段
    for epoch in range(num_epochs):
        total_loss = 0
        total_loss_sum = 0
        epoch_events = []
        total_samples = 0
        model.train()
        # 动态计算对比损失权重
        # contrastive_weight = 0.1 * (1 - epoch / num_epochs)  # 线性衰减
        for batch in pyg_loader:
            # print("Batch type:", type(batch))  # 调试输出
            # print(f"Batch x shape: {batch.x.shape}, Batch edge_index shape: {batch.edge_index.shape}, Batch edge_attr shape: {batch.edge_attr.shape}, Batch y shape: {batch.y.shape}, Batch timestamp shape: {batch.timestamp.shape}")
    
            batch = batch.to(device)  # 将整个批次数据移动到 GPU
           
            optimizer.zero_grad()
            
            # 前向传播：对整个批次图数据进行处理
            predictions = model(batch)
            
            # # 全局平均池化将节点级预测转为图级嵌入（每个图一个嵌入）
            # graph_embeddings = global_mean_pool(predictions, batch.batch)
            
            # 前向传播（现在返回logits和私有特征）
            logits, private_features = model(batch)
            
            # 分类损失
            cls_loss = criterion(logits, batch.y)
            
            # 私有特征对比损失（强制同类靠近、异类远离）
            cont_loss = private_contrastive_loss(private_features, batch.y)
            
            # 总损失
            # total_loss = cls_loss + 0.1*(pos_loss + neg_loss)
            total_loss = cls_loss + 0.1 * cont_loss  # 权重可调
            
            # 总损失计算
            # total_loss = cls_loss + contrastive_weight * (pos_loss + neg_loss)
            # -----------------------------------------
            total_loss.backward()
            optimizer.step()
            
            # total1_loss += total_loss.item() * batch.num_graphs
            # 累加总损失和样本数 
            total_loss_sum += total_loss.item() * batch.num_graphs
            total_samples += batch.num_graphs  # 实际处理的样本数
        
        # 计算平均损失 
        avg_loss = total_loss_sum / total_samples if total_samples > 0 else 0
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        log_metrics_to_file(log_file, epoch, avg_loss)

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            log_file1 = f"{directory_path}/best_model.pth"
            torch.save(model.state_dict(), log_file1)
            print("Best model saved.")
            patience_counter = 0
        else:
            patience_counter += 1


        # 提前停止
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Learning Rate at Epoch {epoch + 1}: {current_lr:.6f}\n")

    # 保存最终模型
    torch.save(model.state_dict(), f"{directory_path}/final_model.pth")
    print("Final model saved.")


    
    
    # 输出显著事件
    print("\n=== 显著边变化事件 ===")
    for event in global_significant_events:
        time_step = event.get('time_step', '未知时间')
        nodes = event.get('nodes', [])
        changes = event.get('change', [])

        if len(nodes) > 0 and len(changes) > 0:
            print(f"时间点: {time_step}")
            for node, change in zip(nodes, changes):
                print(f"  节点对: {node}, 预测类别: {change}")

 

def print_metrics(accuracy, precision, recall, f1, cm, title="Metrics"):
    """
    Prints metrics to the console and logs confusion matrix.
    """
    print(f"\n=== {title} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


def save_confusion_matrix(cm, filename, title):
    """
    Saves confusion matrix as a heatmap image.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(filename)
    plt.close()


def log_metrics_to_file(log_file, epoch, avg_loss):
    """
    Logs epoch metrics to a text file.
    """
    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}\n")


def log_validation_results(log_file, epoch, results):
    """
    Logs validation results (accuracy, precision, recall, F1) to a text file.
    """
    with open(log_file, "a") as f:
        f.write(f"\nEpoch {epoch + 1} - Validation Metrics:\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n")
        f.write(f"Confusion Matrix:\n{results['confusion_matrix']}\n")
        f.write(f"Validation Loss: {results['val_loss']:.4f}\n")

def calculate_classification_loss(predictions_with_pairs, labels, criterion):
    """
    计算三分类任务的损失。
    
    Args:
        predictions: 模型输出的预测概率 (num_edges, 3)
        labels: 真实标签 (batch_size,)
        criterion: 损失函数 (CrossEntropyLoss)
    """
    predictions, node_pairs = predictions_with_pairs[0], predictions_with_pairs[1]
    # 添加数值稳定性检查
    print("Predictions statistics:")
    print("Min:", torch.min(predictions))
    print("Max:", torch.max(predictions))
    print("Has NaN:", torch.isnan(predictions).any())
    
    if torch.isnan(predictions).any():
        print("Warning: NaN values in predictions!")
        # 可以在这里保存导致 NaN 的数据用于调试
    labels = labels - 1
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    labels = labels.long()
    
    # 打印形状信息以便调试
    #print(f"Before processing - Predictions shape: {predictions.shape}, Labels shape: {labels.shape}")
    
    # 将标签映射到0,1,2（因为原始标签是1,2,3）
    #labels = labels - 1
    
    # 移除不在[1,2,3]范围内的样本
    valid_mask = (labels >= 0) & (labels <= 2)
    
    # 确保valid_mask和predictions的第一维度匹配
    if valid_mask.shape[0] != predictions.shape[0]:
        # 如果标签数量与预测数量不匹配，我们需要调整
        # 这里假设我们需要为每个节点对生成一个标签
        num_nodes = int((1 + np.sqrt(1 + 8 * predictions.shape[0])) / 2)
        # 重新构造标签数组以匹配节点对的数量
        new_labels = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                idx = i * num_nodes + j - ((i + 1) * (i + 2)) // 2
                if idx < len(labels):
                    new_labels.append(labels[idx])
                else:
                    new_labels.append(0)  # 对于超出范围的部分使用-1
        
        labels = torch.tensor(new_labels, device=labels.device)
        valid_mask = (labels >= 10) & (labels <= 2)
    
    #print(f"After mask creation - Valid mask shape: {valid_mask.shape}, Predictions shape: {predictions.shape}")
    
    # 应用掩码
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    
    #print(f"After masking - Predictions shape: {predictions.shape}, Labels shape: {labels.shape}")
    
    # 同时过滤节点对
    valid_pairs = [pair for idx, pair in enumerate(node_pairs) if valid_mask[idx]]
    if len(valid_pairs) == 0:
        return torch.tensor(0.0, requires_grad=True), []
    if len(predictions.shape) == 1:
        predictions = predictions.unsqueeze(0)
    
     # 打印调试信息
    print(f"Labels range: {labels.min()}-{labels.max()}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Labels shape: {labels.shape}")
    try:
        loss = criterion(predictions, labels)
    except Exception as e:
        print(f"Error in loss calculation:")
        print(f"Final predictions shape: {predictions.shape}")
        print(f"Final labels shape: {labels.shape}")
        print(f"Unique labels in batch: {torch.unique(labels).numpy()}")
        raise e
        
    return loss,valid_pairs

def convert_classification_predictions_to_edge_features(predictions, num_nodes):
    """
    将分类预测转换为边特征矩阵。
    
    Args:
        predictions: shape (num_edges, num_classes) 的预测概率
        num_nodes: 节点数量
    Returns:
        torch.Tensor: shape (num_nodes, num_nodes) 的边特征矩阵
    """
    # 获取预测的类别（概率最高的类别）
    pred_classes = torch.argmax(predictions, dim=1)+1
    
    # 初始化边特征矩阵
    edge_features = torch.zeros((num_nodes, num_nodes))
    
    # 将预测的类别填充到边特征矩阵中
    for i in range(len(pred_classes)):
        # 假设这里我们需要确定每条边的源节点和目标节点
        # 这部分可能需要根据你的具体实现来修改
        src = i // num_nodes
        dst = i % num_nodes
        edge_features[src, dst] = pred_classes[i].float()
        edge_features[dst, src] = pred_classes[i].float()  # 如果是无向图
        
    return edge_features

def calculate_global_complexity_from_events(events, alpha=1.0, beta=1.0):
    """
    计算分类任务的全局复杂性。
    
    Args:
        events: 事件列表
        alpha: 标准差权重
        beta: 熵权重
    Returns:
        float: 全局复杂性值
    """
    if not events:
        return 0

    # 收集所有类别变化
    changes = []
    for event in events:
        changes.extend(event['change'])

    changes = np.array(changes)
    
    # 计算类别分布
    unique_classes, class_counts = np.unique(changes, return_counts=True)
    total_samples = len(changes)
    
    # 计算熵
    probs = class_counts / total_samples
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    # 计算标准差（基于类别的数值表示）
    std_dev = np.std(changes)
    
    # 计算全局复杂性
    global_complexity = entropy + alpha * std_dev + beta * len(unique_classes)
    
    return global_complexity

# def collate_fn(batch, fs, device):
#     data_list = []
#     for data in batch:
#          # data.x 是原始信号，data.y 是标签
#         new_data = build_graph_data(data.x, fs, device=device)
#         # 如果 build_graph_data 返回的 edge_attr 为 None，则设置默认值
#         if new_data.edge_attr is None:
#             new_data.edge_attr = torch.ones((new_data.edge_index.shape[1], 1), device=device)
#         # 保留原始标签
#         new_data.y = data.y if isinstance(data.y, torch.Tensor) else torch.tensor([data.y], dtype=torch.long)
#         data_list.append(new_data)
#     combined = Batch.from_data_list(data_list)
#     # print("返回的 Batch 类型：", type(combined), flush=True)
#     return combined

