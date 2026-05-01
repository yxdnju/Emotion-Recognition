# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
#
# # 加载数据的函数
# def load_deap_data(filename):
#     with open(filename, 'rb') as f:
#         data = pickle.load(f, encoding='latin1')
#     return data['data']
#
# def detect_scr(signal, height_threshold):
#     """
#     简单的SCR检测方法，基于峰值检测。
#     :param signal: 输入的GSR信号。
#     :param height_threshold: 检测SCR的最小高度阈值。
#     :return: SCR峰值的索引。
#     """
#     # 使用find_peaks方法检测峰值，这里的阈值和距离可能需要根据数据进行调整
#     peaks, _ = find_peaks(signal, height=height_threshold)
#     return peaks
#
# # 数据文件路径，根据你的实际路径调整
# filename = "J:/参考实验/实验/DEAP/data_preprocessed_python/data_preprocessed_python/s01.dat"
# # file_path = 'J:/参考实验/实验/DEAP/data_original/s01.bdf'
#
#
# # 加载数据01
# data = load_deap_data(filename)
#
# # 选择要可视化的信号通道，这里是GSR，位于第37个通道（0索引，所以使用36）
# signal_channel = 37
#
# # 提取第一个视频的GSR信号数据
# signal = data[0][signal_channel]
#
# # 创建时间轴，考虑到采样率是128Hz
# time = np.arange(0, len(signal) / 128, 1/128)
#
# # 绘制GSR信号
# plt.figure(figsize=(14, 7))
# plt.plot(time, signal)
# plt.title('Original Resp  Signal',fontsize=18)
# plt.xlabel('Time (seconds)',fontsize=14)
# plt.ylabel('Resp Signal Value',fontsize=14)
# plt.xticks(fontsize=14)  # Set font size for x ticks
# plt.yticks(fontsize=14)  # Set font size for y ticks
# plt.tight_layout()
# plt.show()

#
# # 检测SCR
# peaks = detect_scr(signal, height_threshold=0.1)  # 阈值设置为0.1，可能需要根据实际信号调整
#
# # 绘制GSR信号和SCR事件
# plt.figure(figsize=(10, 4))
# plt.plot(time, signal, label='GSR Signal')
# plt.plot(time[peaks], signal[peaks], 'rx', label='SCR Events')
# plt.title('Subject 1 - Video 1 - GSR and Detected SCR Signals')
# plt.xlabel('Time (seconds)')
# plt.ylabel('GSR Signal Value')
# plt.legend()
# plt.show()
#





import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# 加载数据的函数
def load_deap_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data['data']

# 简单的低通滤波器
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# 数据文件路径
filename = "J:/参考实验/实验/DEAP/data_preprocessed_python/data_preprocessed_python/s01.dat"

# 加载数据
data = load_deap_data(filename)

# 选择GSR信号通道，去噪，并进行检点
signal_channel = 37
signal = data[0][signal_channel]
fs = 128  # 采样率
cutoff = 0.5  # 我们选择5Hz作为低通滤波的截止频率

# 应用低通滤波器
filtered_signal = butter_lowpass_filter(signal, cutoff, fs)

# 检测峰值
peaks, _ = find_peaks(filtered_signal, prominence=1)  # prominence参数可能需要根据你的数据进行调整

# 创建时间轴
time = np.arange(0, len(signal) / fs, 1/fs)
# 绘制Resp
# 绘制去噪后的GSR信号及其峰值，峰值用红色点表示
plt.figure(figsize=(14, 7))
plt.plot(time, filtered_signal, label='Filtered Resp Signal')  # GSR信号
plt.plot(time[peaks], filtered_signal[peaks], 'rx', label='Peaks')  # 峰值以红色x标记
plt.title('Filtered Resp Signal with Detected Peaks',fontsize=18)
plt.xlabel('Time (seconds)',fontsize=14)
plt.ylabel('Resp Signal Value',fontsize=14)
plt.xticks(fontsize=14)  # Set font size for x ticks
plt.yticks(fontsize=14)  # Set font size for y ticks
plt.legend()
plt.show()




# # # 绘制去噪后的GSR信号及其峰值，峰值用淡红色圆点表示
# # plt.figure(figsize=(15, 5))
# # plt.plot(time, filtered_signal, label='Filtered GSR Signal')  # GSR信号
# # plt.plot(time[peaks], filtered_signal[peaks], 'o', color='#FF8080', label='Peaks')  # 峰值以淡红色圆点标记
# # plt.title('Filtered GSR Signal with Detected Peaks')
# # plt.xlabel('Time (seconds)')
# # plt.ylabel('GSR Signal Value')
# # plt.legend()
# plt.show()




# 绘制去噪后的GSR信号及其峰值，峰值用红色点表示
# plt.figure(figsize=(10, 4))
# plt.plot(time, filtered_signal, label='Filtered GSR Signal')  # GSR信号
# plt.plot(time[peaks], filtered_signal[peaks], 'rx', label='Peaks')  # 峰值以红色x标记
# plt.title('Filtered GSR Signal with Detected Peaks')
# plt.xlabel('Time (seconds)')
# plt.ylabel('GSR Signal Value')
# plt.legend()
# plt.show()

# # 绘制去噪后的GSR信号及其峰值，峰值用红色圆点表示
# plt.figure(figsize=(15, 5))
# plt.plot(time, filtered_signal, label='Filtered GSR Signal')  # GSR信号
# plt.plot(time[peaks], filtered_signal[peaks], 'ro', label='Peaks')  # 峰值以红色圆点标记
# plt.title('Filtered GSR Signal with Detected Peaks')
# plt.xlabel('Time (seconds)')
# plt.ylabel('GSR Signal Value')
# plt.legend()
# plt.show()
