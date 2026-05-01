import scipy.io as sio
import numpy as np
import os

# 文件基础路径
base_path = 'J:/参考实验/实验/SEED/SEED/SEED/ExtractedFeatures/'

# 标签值
labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

# 初始化用于存储所有特征和标签的列表
all_features = []
all_labels = []

# 循环处理每个文件中的每个数据集
for i in range(1, 16):  # 对于每个.mat文件
    file_path = f'{base_path}{i}.1.mat'  # 构造文件路径
    data = sio.loadmat(file_path)  # 加载.mat文件

    for j in range(1, 16):  # 对于每个文件中的de_LDS数据集
        key = f'de_LDS{j}'
        if key in data:
            # 调整数据形状
            feature_data = np.transpose(data[key], (1, 0, 2))  # 将形状由(62, 235, 5)变为(235, 62, 5)
            all_features.append(feature_data)

            # 扩展标签
            label_data = np.full((data[key].shape[1],), labels[j - 1])  # 创建与数据长度相同的标签数组
            all_labels.append(label_data)

# 转换为NumPy数组
all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# 打印形状
print("Feature shape:", all_features.shape)
print("Label shape:", all_labels.shape)

# 保存为Numpy文件，保存地址更新为指定的目录
save_path = 'G:/Experiment/Experiment3/Data/SEEDEach'
if not os.path.exists(save_path):
    os.makedirs(save_path)  # 如果目录不存在，则创建目录

np.save(f'{save_path}/all_de_x.npy', all_features)
np.save(f'{save_path}/all_de_y.npy', all_labels)
