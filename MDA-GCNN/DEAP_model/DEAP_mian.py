import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from model import EEGGraphConvNet
# from model1 import EEGGraphConvNet
# from Mutiel_view.model import *
from DEAP_model.model import *
import logging
import os
import sys
from  CustomDataset import  *
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from SEED_Distance import  *
# import torch.optim.lr_scheduler as lr_scheduler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train_test_split_data(dataset):
    return train_test_split(dataset, test_size=0.2, random_state=42)


def _logger(name, level=logging.INFO, log_directory="G:/Experiment/Experiment3/OutCome/Logger"):
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(os.path.join(log_directory, f"{name}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def compute_test_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.view(-1, 2)  # 确保输出和标签的维度一致
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total

def compute_test_f1_score(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.view(-1, 2)  # 确保输出和标签的维度一致
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return f1_score(all_targets, all_preds, average='weighted')

def train_and_test(model, train_loader, test_loader, criterion, optimizer, scheduler,device, epochs=150):
# def train_and_test(model, train_loader, test_loader, criterion, optimizer, device, epochs=150):

    log_directory = "G:/Experiment/Experiment3/OutCome"
    best_accuracy = 0.0  # 初始化最佳准确率
    logger = _logger("Kfold_th1_alpha1_Mutive_Traing_GCN_Liner")  # 创建日志记录器

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(features)
            # print('output',output.shape)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(output, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        # 测试阶段
        model.eval()
        test_loss = 0
        test_preds, test_labels = [], []
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} Testing"):
                features, labels = features.to(device), labels.to(device)
                output = model(features)
                loss = criterion(output, labels)
                test_loss += loss.item()
                _, preds = torch.max(output, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='weighted')
        test_conf_matrix = confusion_matrix(test_labels, test_preds)

        # 检查是否为最佳模型，如果是则保存
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_path = os.path.join(log_directory, "Save_Model", "Kfold_th1_alpha1_Mutive_best_model.pth")
            best_conf_matrix_path = os.path.join(log_directory, "Confusion_matrix", "kfold_th1_alpha_1_Mutive_best_conf_matrix.npy")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            os.makedirs(os.path.dirname(best_conf_matrix_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            np.save(best_conf_matrix_path, test_conf_matrix)

        # 日志记录
        logger.info(f'Epoch {epoch+1}: Train Loss {train_loss / len(train_loader):.4f}, Train Acc {train_acc:.4f}, '
                    f'Train F1 {train_f1:.4f}, Test Loss {test_loss / len(test_loader):.4f}, '
                    f'Test Acc {test_acc:.4f}, Test F1 {test_f1:.4f}')

        scheduler.step()

# 确保你的模型、数据加载器、优化器等已经定义并传递给这个函数。

# 确保你的模型、数据加载器、优化器等已经定义并传递给这个函数。

# def load_data():
#
#     data_path = 'G:/Experiment/Experiment3/Data/SEED/all_de_x.npy'
#     labels_path = 'G:/Experiment/Experiment3/Data/SEED/all_de_y.npy'
#
#     features = np.load(data_path)
#     labels = np.load(labels_path)
#     labels = labels + 1
#     return features, labels
def get_combined_labels(valence_labels, arousal_labels):
    """
    根据valence和arousal的标签生成组合的四种情绪状态标签。

    :param valence_labels: Valence标签数组，其中0代表低，1代表高。
    :param arousal_labels: Arousal标签数组，其中0代表低，1代表高。
    :return: 组合后的情绪状态标签数组。
    """
    # 计算组合标签
    combined_labels = valence_labels * 2 + arousal_labels

    print(f"Unique combined labels and their counts: {np.unique(combined_labels, return_counts=True)}")
    return combined_labels


def load_deap_data(subject_id, data_dir="G:/Experiment/Experiment3/Data/DEAP"):
    """
    加载单个受试者的数据。
    :param subject_id: 受试者编号，范围从1到32。
    :param data_dir: 数据存储目录。
    :return: 受试者的数据和对应的情绪标签。
    """
    file_path = os.path.join(data_dir, f"DE_S{subject_id}.npz")
    with np.load(file_path) as data:
        features = data['data']  # EEG信号数据
        # valence_labels = data['valence_labels']  # 情绪标签（这里使用valence作为示例）
        arousal_labels = data['arousal_labels']  # 如果需要，也可以使用arousal
        print(arousal_labels)
        # 打印标签值统计信息
        unique, counts = np.unique(arousal_labels, return_counts=True)
        print(f"Unique labels and their counts for subject {subject_id}: {dict(zip(unique, counts))}")
        # combined_labels = get_combined_labels(valence_labels, arousal_labels)
    # 根据实际情况选择使用valence_labels还是arousal_labels
    # return features,combined_labels
    return features,arousal_labels




def main():
    # # 加载数据
    # features, labels = load_data()  # 确保数据是numpy格式
    # # features = features[:25000,:,:5]
    # # labels = labels[:25000]
    # print("Feature shape:", features.shape)
    # print("Label shape:", labels.shape)
    #
    # # 对features和labels应用五折交叉验证
    # kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # 准备记录整体性能
    overall_test_acc = []
    overall_test_f1 = []
    for subject_id in range(1, 33):  # 遍历32个受试者
        print(f"Processing Subject {subject_id}")
        features, labels = load_deap_data(subject_id)
        # features = features[:,:,:2]

        # labels = labels - 1  # 如果标签从1开始，调整为从0开始
        print("Feature shape:", features.shape)
        print("Label shape:", labels.shape)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_ids, test_ids) in enumerate(kfold.split(features),start=1):
            print(f"FOLD {fold}")
            print("--------------------------------")

            # 分割数据
            features_train, labels_train = features[train_ids], labels[train_ids]
            features_test, labels_test = features[test_ids], labels[test_ids]

        # 转换为Tensor
            features_train_tensor = torch.tensor(features_train, dtype=torch.float)
            labels_train_tensor = torch.tensor(labels_train, dtype=torch.long)
            features_test_tensor = torch.tensor(features_test, dtype=torch.float)
            labels_test_tensor = torch.tensor(labels_test, dtype=torch.long)

        # 使用CustomDataset
            train_dataset = CustomDataset(features_train_tensor, labels_train_tensor)
            test_dataset = CustomDataset(features_test_tensor, labels_test_tensor)

        # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

        # 模型和优化器设置
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            electrodes = Electrodes62()  # 确保已经正确实现了这个类
            A_spatial_numpy = electrodes.get_adjacency_matrix()
            A_spatial = torch.tensor(A_spatial_numpy, dtype=torch.float).to(device)
            # print('A_spatial',A_spatial.shape)

            # model = EEGGraphConvNet(num_node_features=4, num_classes=2, A_spatial=A_spatial).to(device)
            # model = EEGGraphConvNet(num_node_features=4, num_classes=2).to(device)  # 参数根据需要调整

            # 假设A_spatial已经定义，并且您已经选择了适当的参数值
            # A_spatial = ...  # 例如，一个torch.Tensor表示的空间邻接矩阵
            # num_node_features = 4  # 每个节点的特征数量
            # num_classes = 2  # 分类任务的类别数
            # nhid1 = 64  # 第一层GCN的隐藏单元数
            # nhid2 = 32  # 第二层GCN的隐藏单元数
            # dropout = 0.5  # dropout比率

            # model = EEGGraphConvNet(num_node_features=4, num_classes=2, nhid1=512, nhid2=256, dropout=0,
            #                         A_spatial=A_spatial).to(device)
            model = EEGGraphConvNet(num_node_features=4, num_classes=2, nhid1=256, nhid2=128, dropout=0,
                                    A_spatial=A_spatial).to(device)

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=1)


        # 训练和测试
            train_and_test(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs=100)
        # train_and_test(model, train_loader, test_loader, criterion, optimizer, device, epochs=100)

        # 这里可以添加代码来记录每一折的性能，例如：
        # overall_test_acc.append(test_acc)
        # overall_test_f1.append(test_f1)

    # 打印整体性能
        print(f"Overall Test Accuracy: {np.mean(overall_test_acc):.4f} ± {np.std(overall_test_acc):.4f}")
        print(f"Overall Test F1 Score: {np.mean(overall_test_f1):.4f} ± {np.std(overall_test_f1):.4f}")

if __name__ == '__main__':
    main()