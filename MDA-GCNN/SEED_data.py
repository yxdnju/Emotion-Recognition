import scipy.io as sio
import numpy as np

# # 路径设置为指定的.mat文件路径
# file_path = 'J:/参考实验/实验/SEED/SEED/SEED/ExtractedFeatures/1.1.mat'
#
# # 加载.mat文件
# data = sio.loadmat(file_path)
#
# # 打印特定键的数据结构和形状
# for i in range(1, 16):
#     key = f'de_LDS{i}'
#     if key in data:
#         print(f"Key: {key}, Shape: {data[key].shape}")
#     else:
#         print(f"Key: {key} not found in the file.")

import scipy.io as sio
import numpy as np

# 文件路径
file_path = 'J:/参考实验/实验/SEED/SEED/SEED/ExtractedFeatures/1.1.mat'

# 加载.mat文件
data = sio.loadmat(file_path)

# 标签值
labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

# 合并数据
all_features = []
all_labels = []

for i in range(1, 16):
    key = f'de_LDS{i}'
    if key in data:
        # 调整数据形状
        feature_data = np.transpose(data[key], (1, 0, 2))  # 将形状由(62, 235, 5)变为(235, 62, 5)
        all_features.append(feature_data)

        # 扩展标签
        label_data = np.full((data[key].shape[1],), labels[i - 1])  # 创建与数据长度相同的标签数组
        all_labels.append(label_data)

# 转换为NumPy数组
all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# 打印形状
print("Feature shape:", all_features.shape)
print("Label shape:", all_labels.shape)

# 保存为Numpy文件
np.save('x.npy', all_features)
np.save('y.npy', all_labels)

# 保存为Numpy文件
np.save('x.npy', all_features)
np.save('y.npy', all_labels)

