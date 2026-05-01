import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from model1 import EEGGraphConvNet
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


def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    preds, labels = [], []
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds.extend(output.argmax(dim=1).tolist())
        labels.extend(data.y.tolist())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return total_loss / len(train_loader), acc, f1


def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    preds, labels = [], []
    with torch.no_grad():
        for data in test_loader:
            output = model(data)
            loss = criterion(output, data.y.view(-1))
            total_loss += loss.item()
            preds.extend(output.argmax(dim=1).tolist())
            labels.extend(data.y.tolist())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return total_loss / len(test_loader), acc, f1

def simulate_data(num_samples=100, num_channels=62, num_features=5, num_classes=3):
    features = np.random.rand(num_samples, num_channels, num_features).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)
    return features, labels


def main():
    features, labels = simulate_data()
    dataset = create_dataset(features, labels)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = EEGGraphConvNet(num_node_features=features.shape[2], num_classes=3)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):  # Assume 10 epochs for simplicity
        train_loss, train_acc, train_f1 = train(model, train_loader, criterion, optimizer)
        test_loss, test_acc, test_f1 = test(model, test_loader, criterion)
        print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Train F1 {train_f1:.4f}, '
              f'Test Loss {test_loss:.4f}, Test Acc {test_acc:.4f}, Test F1 {test_f1:.4f}')

if __name__ == "__main__":
    main()