import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import pickle
from scipy.signal import welch
#
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.fftpack import fft
#
def load_deap_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data['data']
#
def highpass_filter(signal, cutoff=0.5, fs=128, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def detect_scr_peaks(signal, height=None, distance=100):
    peaks, _ = find_peaks(signal, height=height, distance=distance)
    return peaks

#时域特征
def calculate_scr_features(filtered_signal, peaks):
    peak_amplitudes = filtered_signal[peaks] if peaks.size > 0 else np.array([])
    features = {

        'max_amplitude': np.max(peak_amplitudes) if peak_amplitudes.size > 0 else 0,
        'mean_amplitude': np.mean(peak_amplitudes) if peak_amplitudes.size > 0 else 0,
        'peak_count': len(peaks),
        'min_amplitude': np.min(peak_amplitudes) if peak_amplitudes.size > 0 else 0,
        'mean_value': np.mean(filtered_signal),
    }
    return features
#频域特征
# 计算PSD特征的函数
def calculate_psd_features(data, signal_channel, stft_para):
    STFTN = stft_para['stftn']
    fStart = stft_para['fStart']
    fEnd = stft_para['fEnd']
    fs = stft_para['fs']
    window = stft_para['window']
    m = data.shape[2]  # 获取每个信号的样本数
    Hwindow = np.hanning(m)

    psd_features = []
    for i in range(data.shape[0]):
        temp = data[i, signal_channel, :]
        filtered_signal = highpass_filter(temp, cutoff=0.05, fs=128, order=5)
        temp = filtered_signal
        Hdata = temp * Hwindow
        FFTdata = fft(Hdata)
        magFFTdata = abs(FFTdata[0:int(m / 2)])

        psd_feature = {}
        for p, (start, end) in enumerate(zip(fStart, fEnd)):
            fStartNum = int(start / fs * m)
            fEndNum = int(end / fs * m)
            E = np.sum(magFFTdata[fStartNum:fEndNum] ** 2) / (fEndNum - fStartNum + 1)
            psd_feature[f'band{p}_power'] = E
        psd_features.append(psd_feature)
    return psd_features


# 数据文件路径，根据你的实际路径调整
filename = "J:/参考实验/实验/DEAP/data_preprocessed_python/data_preprocessed_python/s01.dat"


# 加载数据
data = load_deap_data(filename)

# 选择GSR信号通道，位于第37个通道（0索引，所以使用36）
signal_channel = 36

all_features = []
# 对每个视频的GSR信号进行处理并计算PSD特征
all_psd_features = []

# 初始化STFT参数
stft_para = {
    'stftn': 512,  # 修改为FFT点数
    'fStart': [0.05, 0.14, 0.23, 0.32, 0.41],
    # 'fStart': [0.04, 0.15],
    #
    # 'fEnd': [0.15, 0.4],
    'fEnd': [0.14, 0.23, 0.32, 0.41, 0.50],

    'window': 1,  # 窗口长度，以秒为单位
    'fs': 128  # 原始频率
}

all_features = []

# 对每个视频的GSR信号进行处理并计算特征
for i in range(data.shape[0]):
    signal = data[i, signal_channel, :]
    filtered_signal = highpass_filter(signal, cutoff=0.05, fs=128, order=5)
    peaks = detect_scr_peaks(filtered_signal, distance=100)
    scr_features = calculate_scr_features(filtered_signal, peaks)
    all_features.append(scr_features)

psd_features = calculate_psd_features(data, signal_channel, stft_para)

# 打印特征结果
for video_index, (scr_features, psd_feature) in enumerate(zip(all_features, psd_features)):
    print(f"Video {video_index + 1}:")
    for feature, value in scr_features.items():
        print(f"  {feature}: {value}")
    for feature, value in psd_feature.items():
        print(f"  {feature}: {value}")

