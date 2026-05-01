import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# from model import EEGGraphConvNet
from model1 import EEGGraphConvNet

import os
from  CustomDataset import  *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




def create_dataset(features, labels):
    dataset = []
    for feature, label in zip(features, labels):
        # 这里需要您根据自己的数据调整
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # 示例的边索引
        x = torch.tensor(feature, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
    return dataset


def train_test_split_data(dataset):
    return train_test_split(dataset, test_size=0.2, random_state=42)



# def train_and_test(model, train_loader, test_loader, criterion, optimizer, epochs=10):
def train_and_test(model, train_loader, test_loader, criterion, optimizer, device, epochs=100):
    for epoch in range(epochs):
        model.train()
        train_loss, train_preds, train_labels = 0, [], []
        # for batch in train_loader:
        # for data in train_loader:
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            # features, labels = batch
            # print('before',features.shape)

            optimizer.zero_grad()
            # output = model(data)
            # output = model(data['x'])
            output = model(features)
            loss = criterion(output,labels)
            # loss = criterion(output, data['y'].view(-1))
            # print('output',output.shape)
            # loss = criterion(output, data.y.view(-1))
            # loss = criterion(output, labels.view(-1))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend(output.argmax(dim=1).cpu().tolist())
            train_labels.extend(labels.cpu().tolist())

        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        model.eval()
        test_loss, test_preds, test_labels = 0, [], []
        with torch.no_grad():
            # for data in test_loader:
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)

                output = model(features)
                loss = criterion(output, labels)
                test_loss += loss.item()
                test_preds.extend(output.argmax(dim=1).cpu().tolist())
                test_labels.extend(labels.cpu().tolist())

        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='weighted')

        print(f'Epoch {epoch}: Train Loss {train_loss / len(train_loader):.4f}, Train Acc {train_acc:.4f}, '
              f'Train F1 {train_f1:.4f}, Test Loss {test_loss / len(test_loader):.4f}, '
              f'Test Acc {test_acc:.4f}, Test F1 {test_f1:.4f}')

def simulate_data(num_samples=100, num_channels=62, num_features=5, num_classes=3):
    features = np.random.rand(num_samples, num_channels, num_features).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)
    return features, labels


def load_data():
    # 替换为您的文件路径
    # data_path = 'path_to_your_data.npy'
    # labels_path = 'path_to_your_labels.npy'
    data_path = 'G:/Experiment/Experiment3/x.npy'
    labels_path = 'G:/Experiment/Experiment3/y.npy'

    features = np.load(data_path)
    labels = np.load(labels_path)
    labels = labels + 1
    return features, labels

def main():
    # 模拟数据加载、模型初始化、数据集创建和加载器准备等逻辑
    # features, labels = simulate_data()
    features, labels = load_data()
    print("Feature shape:", features.shape)
    print("Label shape:", labels.shape)

    # dataset = create_dataset(features, labels)
    # Convert to tensors
    features_tensor = torch.tensor(features, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    # Split data into training and testing sets

    features_train, features_test, labels_train, labels_test = train_test_split(
        features_tensor, labels_tensor, test_size=0.2, random_state=42)

    # train_loader = DataLoader(list(zip(features_train, labels_train)), batch_size=32, shuffle=True)
    # test_loader = DataLoader(list(zip(features_test, labels_test)), batch_size=32, shuffle=False)

    # 使用CustomDataset
    train_dataset = CustomDataset(features_train, labels_train)
    test_dataset = CustomDataset(features_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # for features, labels in train_loader:
    #     print(features.shape)  # 预期输出: torch.Size([32, 62, 5])
    #     print(labels.shape)  # 预期输出: torch.Size([32])

        # 检查CUDA是否可用，并据此设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        # print("Using device:", device)


    model = EEGGraphConvNet(num_node_features=5, num_classes=3).to(device)  # 参数根据需要调整
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # train_and_test(model, train_loader, test_loader, criterion, optimizer)
    # Proceed with training and testing
    # train_and_test(model, train_loader, test_loader, criterion, optimizer)
    # 将device作为参数传递
    train_and_test(model, train_loader, test_loader, criterion, optimizer, device)

if __name__ == '__main__':
    main()