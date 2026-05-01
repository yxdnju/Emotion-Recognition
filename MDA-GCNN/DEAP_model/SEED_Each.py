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
    train_dataset_size = len(train_loader.dataset)
    test_dataset_size = len(test_loader.dataset)
    print(
        f"Subject {current_subject}: Training set size: {train_dataset_size}, Test set size: {test_dataset_size}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        for features, labels in tqdm(train_loader, desc=f"Subject {current_subject}, Epoch {epoch+1}/{epochs}, Training"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(output, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        model.eval()
        test_loss = 0
        test_preds, test_labels = [], []
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc=f"Subject {current_subject}, Epoch {epoch+1}/{epochs}, Testing"):
                features, labels = features.to(device), labels.to(device)
                output = model(features)
                loss = criterion(output, labels)
                test_loss += loss.item()
                _, preds = torch.max(output, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='weighted')

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_path = os.path.join(log_directory, "Save_Model", f"SEED_EACH_best_model_subject_{current_subject}.pth")
            best_conf_matrix_path = os.path.join(log_directory, "Confusion_matrix", f"SEED_EACH_best_conf_matrix_subject_{current_subject}.npy")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            os.makedirs(os.path.dirname(best_conf_matrix_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            np.save(best_conf_matrix_path, confusion_matrix(test_labels, test_preds))

        logger.info(f'Subject {current_subject}, Epoch {epoch+1}: Train Loss {train_loss / len(train_loader):.4f}, Train Acc {train_acc:.4f}, '
                    f'Train F1 {train_f1:.4f}, Test Loss {test_loss / len(test_loader):.4f}, Test Acc {test_acc:.4f}, Test F1 {test_f1:.4f}')

        scheduler.step()



def load_subject_data(subject_id, data_dir='G:/Experiment/Experiment3/Data/SEEDEach'):
    features_path = os.path.join(data_dir, f'Subject_{subject_id}', 'features.npy')
    labels_path = os.path.join(data_dir, f'Subject_{subject_id}', 'labels.npy')
    features = np.load(features_path)
    labels = np.load(labels_path)
    return features, labels



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_subjects = 15
    epochs = 50
    logger = _logger("SEED_EACH_Training")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for epoch in range(epochs):
        for subject_id in range(1, num_subjects + 1):
            features, labels = load_subject_data(subject_id)
            labels = labels.astype(np.long) + 1
            dataset_size = len(labels)
            logger.info(f"Processing Subject {subject_id} with dataset size: {dataset_size}")

            features = torch.tensor(features, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.long)

            for fold, (train_indices, test_indices) in enumerate(kfold.split(features), 1):
                # logger.info(f"Starting Fold {fold} for Subject {subject_id}")
                # logger.info(f"Starting Epoch {epoch + 1}, Fold {fold} for Subject {subject_id}")
                features_train, features_test = features[train_indices], features[test_indices]
                labels_train, labels_test = labels[train_indices], labels[test_indices]

                train_dataset = CustomDataset(features_train, labels_train)
                test_dataset = CustomDataset(features_test, labels_test)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # 创建模型、优化器、调度器
            electrodes = Electrodes62()  # 假设这是您的电极配置工具
            A_spatial_numpy = electrodes.get_adjacency_matrix()
            A_spatial = torch.tensor(A_spatial_numpy, dtype=torch.float).to(device)
            # print('A_spatial.shape',A_spatial.shape)

            # 创建模型、优化器、调度器
            model = EEGGraphConvNet(num_node_features=5, num_classes=3, nhid1=512, nhid2=256, dropout=0,
                                    A_spatial=torch.tensor(A_spatial, dtype=torch.float)).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

            # 训练和验证
            train_and_test(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs, logger, subject_id)

            # logger.info(f"Fold {fold} completed for Subject {subject_id}.")

    # logger.info("All subjects and folds have been processed.")


if __name__ == '__main__':
    main()



