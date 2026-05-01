####################分离SCR#####################
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pickle

# 加载数据的函数
def load_deap_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data['data']

def highpass_filter(signal, cutoff=0.5, fs=128, order=5):
    """
    高通滤波器，用于强调SCR相关的快速变化。
    :param signal: 输入的GSR信号。
    :param cutoff: 高通滤波器的截止频率。
    :param fs: 采样率。
    :param order: 滤波器的阶数。
    :return: 过滤后的信号。
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# 数据文件路径，根据你的实际路径调整
filename = "J:/参考实验/实验/DEAP/data_preprocessed_python/data_preprocessed_python/s01.dat"

# 加载数据
data = load_deap_data(filename)

# 选择要可视化的信号通道，这里是GSR，位于第37个通道（0索引，所以使用36）
signal_channel = 36

# 提取第一个视频的GSR信号数据
signal = data[0][signal_channel]

# 高通滤波以强调SCR
filtered_signal = highpass_filter(signal, cutoff=0.05)  # 0.05Hz作为截止频率

# 创建时间轴，考虑到采样率是128Hz
time = np.arange(0, len(signal) / 128, 1/128)

# 绘制原始GSR信号
plt.figure(figsize=(14, 7))
plt.plot(time, signal)
plt.title('Original GSR Signal',fontsize=18)
plt.xlabel('Time (seconds)',fontsize=14)
plt.ylabel('GSR Signal Value',fontsize=14)
plt.xticks(fontsize=14)  # Set font size for x ticks
plt.yticks(fontsize=14)  # Set font size for y ticks
plt.show()
#
# 绘制处理后的SCR信号
plt.figure(figsize=(14, 7))
plt.plot(time, filtered_signal)
plt.title('Filtered Signal Emphasizing SCR',fontsize=18)
plt.xlabel('Time (seconds)',fontsize=14)
plt.ylabel('Filtered Signal Value',fontsize=14)
plt.xticks(fontsize=14)  # Set font size for x ticks
plt.yticks(fontsize=14)  # Set font size for y ticks
plt.show()
# #
# #############################这段代码使用detect_scr_peaks函数来检测处理后的SCR信号中的峰值，
# 然后在同一张图上绘制了这些峰值（作为红色的'x'标记）和SCR信号。distance参数用于控制检测到的峰值之间的最小距离，
# 而height参数可以设置峰值的最小高度，这里我们没有设置height，但你可以根据你的信号特性进行调整以过滤掉不需要的峰值。
# 这样的可视化有助于分析SCR活动及其对应的生理反应。
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import pickle

# 加载数据的函数
def load_deap_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data['data']

def highpass_filter(signal, cutoff=0.5, fs=128, order=5):
    """
    高通滤波器，用于强调SCR相关的快速变化。
    :param signal: 输入的GSR信号。
    :param cutoff: 高通滤波器的截止频率。
    :param fs: 采样率。
    :param order: 滤波器的阶数。
    :return: 过滤后的信号。
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# 检测SCR检点
def detect_scr_peaks(signal, height=None, distance=100):
    """
    检测SCR信号中的峰值。
    :param signal: 输入的SCR信号。
    :param height: 峰值的最小高度，用于过滤较小的峰值。
    :param distance: 相邻峰值之间的最小样本数，用于避免检测到非常接近的峰值。
    :return: 峰值的索引。
    """
    peaks, _ = find_peaks(signal, height=height, distance=distance)
    return peaks

# 数据文件路径，根据你的实际路径调整
filename = "J:/参考实验/实验/DEAP/data_preprocessed_python/data_preprocessed_python/s01.dat"

# 加载数据
data = load_deap_data(filename)

# 选择要可视化的信号通道，这里是GSR，位于第37个通道（0索引，所以使用36）
signal_channel = 36

# 提取第一个视频的GSR信号数据
signal = data[0][signal_channel]

# 高通滤波以强调SCR
filtered_signal = highpass_filter(signal, cutoff=0.05)  # 0.05Hz作为截止频率

# 检测SCR检点
peaks = detect_scr_peaks(filtered_signal, height=None, distance=100)  # 参数可能需要根据信号调整

# 创建时间轴，考虑到采样率是128Hz
time = np.arange(0, len(signal) / 128, 1/128)

# 绘制处理后的SCR信号及其检点
plt.figure(figsize=(14, 7))
plt.plot(time, filtered_signal, label='Filtered SCR Signal')
plt.plot(time[peaks], filtered_signal[peaks], 'rx', label='SCR Peaks', markersize=4)  # 将markersize设置为较小的值
plt.title('Filtered SCR Signal with Detected Peaks',fontsize=18)
plt.xlabel('Time (seconds)',fontsize=14)
plt.ylabel('Filtered SCR Value',fontsize=14)
plt.xticks(fontsize=14)  # Set font size for x ticks
plt.yticks(fontsize=14)  # Set font size for y ticks
plt.legend()
plt.show()

