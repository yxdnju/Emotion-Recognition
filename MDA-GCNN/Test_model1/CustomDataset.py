from torch.utils.data import Dataset, DataLoader
#
# class CustomDataset(Dataset):
#     def __init__(self, features, labels):
#         self.features = features
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         return {'x': self.features[idx], 'y': self.labels[idx]}


from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        # self.features = torch.tensor(features, dtype=torch.float)  # 确保特征为Tensor
        # self.labels = torch.tensor(labels, dtype=torch.long)  # 确保标签为Tensor

        self.features = features.clone().detach().float()  # 假设features已经是一个张量
        self.labels = labels.clone().detach().long()  # 假设labels已经是一个张量


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
