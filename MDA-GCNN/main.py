import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# from model import EEGGraphConvNet
from model import EEGGraphConvNet

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_data():
    # 替换为您的文件路径
    data_path = 'path_to_your_data.npy'
    labels_path = 'path_to_your_labels.npy'

    features = np.load(data_path)
    labels = np.load(labels_path)

    return features, labels


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



def train_and_test(model, train_loader, test_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss, train_preds, train_labels = 0, [], []
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            print('output',output.shape)
            loss = criterion(output, data.y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend(output.argmax(dim=1).tolist())
            train_labels.extend(data.y.tolist())

        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        model.eval()
        test_loss, test_preds, test_labels = 0, [], []
        with torch.no_grad():
            for data in test_loader:
                output = model(data)
                loss = criterion(output, data.y.view(-1))
                test_loss += loss.item()
                test_preds.extend(output.argmax(dim=1).tolist())
                test_labels.extend(data.y.tolist())

        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average='weighted')

        print(f'Epoch {epoch}: Train Loss {train_loss / len(train_loader):.4f}, Train Acc {train_acc:.4f}, '
              f'Train F1 {train_f1:.4f}, Test Loss {test_loss / len(test_loader):.4f}, '
              f'Test Acc {test_acc:.4f}, Test F1 {test_f1:.4f}')

def simulate_data(num_samples=100, num_channels=62, num_features=5, num_classes=3):
    features = np.random.rand(num_samples, num_channels, num_features).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)
    return features, labels

def main():
    # 模拟数据加载、模型初始化、数据集创建和加载器准备等逻辑
    features, labels = simulate_data()
    dataset = create_dataset(features, labels)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = EEGGraphConvNet(num_node_features=5, num_classes=3)  # 参数根据需要调整
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_and_test(model, train_loader, test_loader, criterion, optimizer)
if __name__ == '__main__':
    main()