import mne
import matplotlib.pyplot as plt

# 定义文件路径
file_path = 'J:/参考实验/实验/DEAP/data_original/s01.bdf'

# 读取.bdf文件
raw = mne.io.read_raw_bdf(file_path, preload=True)

# 获取数据和时间信息
data, times = raw[20, :]

# 绘制第41个通道的数据
plt.figure(figsize=(15, 5))
plt.plot(times, data.T)
plt.title('Channel 41 Data')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.show()
