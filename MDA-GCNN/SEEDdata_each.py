import scipy.io as sio
import numpy as np
import os

# 文件基础路径
base_path = 'J:/参考实验/实验/SEED/SEED/SEED/ExtractedFeatures/'

# 标签值
labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

# 保存数据的路径
save_path_base = 'G:/Experiment/Experiment3/Data/SEEDEach'

# 确保保存路径存在
if not os.path.exists(save_path_base):
    os.makedirs(save_path_base)

# 循环处理每个受试者的数据
for i in range(1, 16):  # 对于每个受试者
    subject_features = []  # 初始化用于存储当前受试者特征的列表
    subject_labels = []  # 初始化用于存储当前受试者标签的列表

    # 构造当前受试者的.mat文件路径
    file_path = os.path.join(base_path, f'{i}.1.mat')
    data = sio.loadmat(file_path)  # 加载.mat文件

    # 处理当前受试者的每个de_LDS数据集
    for j in range(1, 16):
        key = f'de_LDS{j}'
        if key in data:
            # 调整数据形状为(235, 62, 5)
            feature_data = np.transpose(data[key], (1, 0, 2))
            subject_features.append(feature_data)

            # 创建与特征数据长度相同的标签数组
            label_data = np.full((feature_data.shape[0],), labels[j - 1])
            subject_labels.append(label_data)

    # 转换列表为NumPy数组
    subject_features = np.concatenate(subject_features, axis=0)
    subject_labels = np.concatenate(subject_labels, axis=0)

    # 为当前受试者创建保存文件的完整路径
    save_path_subject = os.path.join(save_path_base, f'Subject_{i}')
    if not os.path.exists(save_path_subject):
        os.makedirs(save_path_subject)  # 如果目录不存在，则创建目录

    # 保存当前受试者的特征和标签到.npy文件
    np.save(os.path.join(save_path_subject, 'features.npy'), subject_features)
    np.save(os.path.join(save_path_subject, 'labels.npy'), subject_labels)

    print(f'Subject {i} data saved.')
