import os
import pickle
import numpy as np
import torch
import gc
import random
# from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from Dataset import CustomDataset
from sklearn.model_selection import KFold  # 引入 KFold
from sklearn.utils import shuffle
from dynamic_gnn import DynamicGNN, TemporalGNN

from train_and_detect import train_dynamic_gnn_with_events, evaluate_dynamic_gnn
from extract_features_graph_builder import build_graph_data
import CustomLoss


def reset_environment():
    # 清空 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 删除所有变量并运行垃圾回收
    gc.collect()

    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Environment reset completed.")    

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reset_environment()

# 定义数据文件路径
data_dir = "D:/pythonwork/project/data"
data_files = [f"filtered_aligned_chest_data_250Hz_S{i}.pkl" for i in range(2, 18) if i != 12]  # S2.pkl 到 S17.pkl

# 定义窗口参数
window_size = 5  # 窗口大小（秒）
stride = 2  # 滑动步长（秒）
sampling_rate = 250  # 采样率（Hz）
samples_per_window = int(window_size * sampling_rate)  # 每个窗口的样本数
stride_samples = int(stride * sampling_rate)  # 滑动步长的样本数

# 定义有效标签和映射
valid_labels = [1, 2, 3]
label_mapping = {1: 0, 2: 1, 3: 2}

# 定义批大小和 DataLoader 参数
batch_size = 64
num_workers = 8

 # 加载所有数据
all_signals = []
all_labels = []
all_timestamps = []
all_graph_data = []
  # 选择的信号通道
# for file in data_files:
#     subject_id = int(file.split('_S')[1].split('.')[0])  # 例如从文件名提取
#     signals, labels, timestamps = load_and_preprocess_data(os.path.join(data_dir, file))
#     for idx in range(len(signals)):
#         # 调用 build_graph_data 构建图数据
#         data = build_graph_data(signals[idx], fs=sampling_rate, device=device)  # 假设z已经在函数内部处理
        
#         # 如果 edge_attr 为 None，则赋予默认值
#         if data.edge_attr is None:
#             data.edge_attr = torch.ones((data.edge_index.shape[1], 1), device=device)
        
#         # 转换标签为 tensor 并赋值给 data.y
#         label_tensor = torch.tensor([labels[idx]], dtype=torch.long)
#         data.y = label_tensor  # 保存标签
        
#         data.subject_id = torch.tensor([subject_id], dtype=torch.long, device=device)
#         data.timestamp = torch.tensor([timestamps[idx][0]], dtype=torch.float32, device=device)

#         # 将生成的图数据添加到列表中
#         all_graph_data.append(data)
# # # 训练和评估
directory_path = "D:/pythonwork/project/result/ablation"
# 保存图数据集
# torch.save(all_graph_data, os.path.join(directory_path, "graphs_data2_5sec.pt"))
# print("图数据集已保存.")

# 读取图数据集并进行十折交叉验证
load_existing = True  
all_graphs = torch.load("D:/pythonwork/project/result/ablation/graphs_data2_5sec.pt")
# all_graphs = torch.load("D:/pythonwork/project/3.10.produce/nurse/graphs_data2_nurse_20000_10sec.pt")


n_splits = 10
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
labels_for_split = [graph.y.item() for graph in all_graphs]
# 如果需要放到 GPU，可以逐个将图数据转移到 GPU，例如：
all_graphs = [data.to(device) for data in all_graphs]



print(f"总窗口样本数：{len(all_graphs)}")

for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(all_graphs)), np.array([graph.y.item() for graph in all_graphs]))):
    # print(f"\n=== Fold {fold+1}/{n_splits} ===")
    # if fold != 6:  # 仅处理第七折（enumerate从0开始，第七折是索引6）
    #     continue
    
    # print(f"\n=== 单独运行第七折 ===")
    # 划分训练集和验证集（图数据列表）
    train_graphs = [all_graphs[i] for i in train_idx]
    test_graphs = [all_graphs[i] for i in test_idx]
    
    # 打印标签分布（可选：遍历图数据提取标签）
    train_labels = np.array([graph.y.item() for graph in train_graphs])
    test_labels = np.array([graph.y.item() for graph in test_graphs])
    print(f"Training set - Labels distribution: {np.bincount(train_labels)}")
    print(f"Validation set - Labels distribution: {np.bincount(test_labels)}")
    
   
    # 对训练集进行增强
    # train_graphs = augment_dataset(train_graphs, noise_level=0.005)
    # 创建 DataLoader（数据已经是图对象，无需额外 collate_fn）
    # train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # 初始化模型并迁移到设备
    # model = DynamicGNN(input_dim, hidden_dim, output_dim, device).to(device)
    model = TemporalGNN(
    input_dim=14, 
    hidden_dim=128,
    output_dim=128,
    device=device
)
    
    results = train_dynamic_gnn_with_events(model, train_graphs, directory_path, batch_size=batch_size, fs=sampling_rate, device=device)

    # 加载最佳模型
    model.load_state_dict(torch.load(f"{directory_path}/best_model.pth"))
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=4)

    # 评估模型
    test_results = evaluate_dynamic_gnn(model, test_graphs, criterion, fs=sampling_rate, device=device)

    # 保存结果
    log_file = f"{directory_path}/evaluation_log.txt"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n=== Fold {fold + 1} 测试集性能 ===\n")
        f.write(f"Accuracy: {test_results['accuracy']:.4f}\n")
        f.write(f"Precision: {test_results['precision']:.4f}\n")
        f.write(f"Recall: {test_results['recall']:.4f}\n")
        f.write(f"F1 Score: {test_results['f1']:.4f}\n")
        f.write(f"cm: {test_results['cm']}\n")

    print(f"Fold {fold + 1} completed.")
