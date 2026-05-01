import os
import numpy as np
import math
import scipy.io as sio
from scipy.fftpack import fft
import pickle
from scipy.signal import butter, filtfilt, welch


def highpass_filter(signal, cutoff=0.5, fs=128, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def load_deap_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data['data']

filename = "J:/参考实验/实验/DEAP/data_preprocessed_python/data_preprocessed_python/s28.dat"
data = load_deap_data(filename)
signal_channel = 36  # 第37个通道

# 初始化STFT参数
stft_para = {
    'stftn': 4,  # 频域采样率
    'fStart': [0.05, 0.14, 0.23, 0.32, 0.41],  # 起始频率
    'fEnd': [0.14, 0.23, 0.32, 0.41, 0.50],  # 结束频率
    'window': 1,  # 窗口长度，以秒为单位
    'fs': 128  # 原始频率
}

def DE_PSD(data, stft_para):
    '''
    计算 PSD
    --------
    输入:  data [n*m]          n 电极, m 时间点
           stft_para          STFT 参数
    输出: psd [n*k]           n 电极, k 频带
    '''
    STFTN = stft_para['stftn']
    fStart = stft_para['fStart']
    fEnd = stft_para['fEnd']
    fs = stft_para['fs']
    window = stft_para['window']

    fStartNum = np.zeros([len(fStart)], dtype=int)
    fEndNum = np.zeros([len(fEnd)], dtype=int)
    for i in range(len(fStart)):
        fStartNum[i] = int(fStart[i] / fs * STFTN)
        fEndNum[i] = int(fEnd[i] / fs * STFTN)

    n = data.shape[0]
    m = data.shape[2]  # 获取每个信号的样本数

    # 更新汉宁窗以匹配信号长度
    Hwindow = np.hanning(m)  # 使用 np.hanning 生成适当长度的汉宁窗

    psd = np.zeros([data.shape[0], len(fStart)])  # 对每个实验进行修改
    for i in range(data.shape[0]):  # 循环每个实验
        temp = data[i, signal_channel, :]  # 选择第37个通道
        filtered_signal = highpass_filter(temp, cutoff=0.05, fs=128, order=5)
        temp =  filtered_signal
        Hdata = temp * Hwindow
        FFTdata = fft(Hdata, m)  # 使用信号长度进行 FFT
        magFFTdata = abs(FFTdata[0:int(m / 2)])  # 调整为使用 m/2
        for p in range(len(fStart)):
            E = 0
            for p0 in range(fStartNum[p] - 1, fEndNum[p]):
                E += magFFTdata[p0] ** 2
            E /= (fEndNum[p] - fStartNum[p] + 1)
            psd[i][p] = E

    return psd


# 计算PSD特征
psd_features = DE_PSD(data, stft_para)
# 打印第37个通道的PSD特征
print("PSD features for the 37th channel across all experiments:")
print(psd_features)
