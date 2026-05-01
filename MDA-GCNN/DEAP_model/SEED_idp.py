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

def train_and_test(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs, logger, current_subject):
    best_accuracy = 0.0
    log_directory = "G:/Experiment/Experiment3/OutCome"
    best_conf_matrix = None  # 用于保存最佳混淆矩阵

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, labels in tqdm(train_loader, desc=f"Subject {current_subject}, Epoch {epoch+1}/{epochs}, Training"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 测试阶段
        model.eval()
        test_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc=f"Subject {current_subject}, Epoch {epoch+1}/{epochs}, Testing"):
                features, labels = features.to(device), labels.to(device)
                output = model(features)
                loss = criterion(output, labels)
                test_loss += loss.item()
                _, preds = torch.max(output, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average='weighted')
        current_conf_matrix = confusion_matrix(all_labels, all_preds)

        logger.info(f'Subject {current_subject}, Epoch {epoch+1}: Train Loss {train_loss / len(train_loader):.4f}, Test Loss {test_loss / len(test_loader):.4f}, Test Acc {test_acc:.4f}, Test F1 {test_f1:.4f}')

        # 如果当前模型的测试准确率超过了之前的最佳准确率，保存模型和混淆矩阵
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_conf_matrix = current_conf_matrix
            best_model_path = os.path.join(log_directory, "Save_Model", f"idp_best_model_subject_{current_subject}.pth")
            best_conf_matrix_path = os.path.join(log_directory, "Confusion_matrix", f"idp_best_conf_matrix_subject_{current_subject}.npy")
            torch.save(model.state_dict(), best_model_path)
            np.save(best_conf_matrix_path, best_conf_matrix)

        scheduler.step()

    # 最终，记录该受试者的最高准确率和相应的混淆矩阵
    logger.info(f'Subject {current_subject}: Best Test Acc {best_accuracy:.4f}')




def load_subject_data(subject_id, data_dir='G:/Experiment/Experiment3/Data/SEEDEach'):
    features_path = os.path.join(data_dir, f'Subject_{subject_id}', 'features.npy')
    labels_path = os.path.join(data_dir, f'Subject_{subject_id}', 'labels.npy')
    features = np.load(features_path)
    labels = np.load(labels_path)
    return features, labels



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_subjects = 15
    epochs = 100  # 每个受试者的训练周期
    logger = _logger("SEED_idp_Training")

    # 加载所有受试者的数据
    all_features = []
    all_labels = []
    for subject_id in range(1, num_subjects + 1):
        features, labels = load_subject_data(subject_id)
        all_features.append(features)
        all_labels.append(labels.astype(np.long) + 1)  # 调整标签
        logger.info(f"Loaded data for Subject {subject_id} with shape: Features {features.shape}, Labels {labels.shape}")

    # 初始化模型、优化器、调度器
    electrodes = Electrodes62()  # 已经实现
    A_spatial_numpy = electrodes.get_adjacency_matrix()
    A_spatial = torch.tensor(A_spatial_numpy, dtype=torch.float).to(device)
    model = EEGGraphConvNet(num_node_features=5, num_classes=3, nhid1=512, nhid2=256, dropout=0, A_spatial=A_spatial).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    # 对每个受试者进行处理，将其作为测试集，其他所有受试者作为训练集
    for test_subject_id in range(num_subjects):
        logger.info(f"\nProcessing Subject {test_subject_id+1} as test set.")

        # 准备测试集
        test_features = all_features[test_subject_id]
        test_labels = all_labels[test_subject_id]
        test_dataset = CustomDataset(torch.tensor(test_features, dtype=torch.float), torch.tensor(test_labels, dtype=torch.long))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 准备训练集
        train_features = np.concatenate([all_features[i] for i in range(num_subjects) if i != test_subject_id], axis=0)
        train_labels = np.concatenate([all_labels[i] for i in range(num_subjects) if i != test_subject_id], axis=0)
        train_dataset = CustomDataset(torch.tensor(train_features, dtype=torch.float), torch.tensor(train_labels, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        logger.info(f"Training set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

        # 训练和测试
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}, Training with Subject {test_subject_id+1} as test set.")
            train_and_test(model, train_loader, test_loader, criterion, optimizer, scheduler, device, 1, logger, test_subject_id + 1)
            scheduler.step()

    logger.info("Training and testing on all subjects completed.")

if __name__ == '__main__':
    main()





