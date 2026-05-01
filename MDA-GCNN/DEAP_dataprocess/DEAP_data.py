import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def read_file(file):
    data = sio.loadmat(file)
    data = data['data']
    print(data.shape)
    return data


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


def decompose(file):
    # trial*channel*sample
    start_index = 384  # 3s pre-trial signals
    data = read_file(file)
    shape = data.shape
    frequency = 128

    # decomposed_de = np.empty([0, 4, 120])
    decomposed_de = np.empty([0, 5, 240])  #5个频段

    base_DE = np.empty([0, 160])

    for trial in range(40):
        temp_base_DE = np.empty([0])
        temp_base_theta_DE = np.empty([0])
        temp_base_alpha_DE = np.empty([0])
        temp_base_beta_DE = np.empty([0])
        temp_base_gamma_DE = np.empty([0])
        temp_base_delta_DE = np.empty([0])  # 添加δ波

        temp_de = np.empty([0, 240])

        for channel in range(32):
            trial_signal = data[trial, channel, 384:]
            base_signal = data[trial, channel, :384]
            # ****************compute base DE****************
            base_delta = butter_bandpass_filter(base_signal, 1, 4, frequency, order=3)
            base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)
            base_alpha = butter_bandpass_filter(base_signal, 8, 13, frequency, order=3)
            base_beta = butter_bandpass_filter(base_signal, 13, 30, frequency, order=3)
            base_gamma = butter_bandpass_filter(base_signal, 30, 50, frequency, order=3)

            # # 计算δ波的基础DE值
            base_delta_DE = (compute_DE(base_delta[:64]) + compute_DE(base_delta[64:128]) + compute_DE(
                base_delta[128:192]) + compute_DE(base_delta[192:256]) + compute_DE(base_delta[256:320]) + compute_DE(
                base_delta[320:])) / 6

            base_theta_DE = (compute_DE(base_theta[:64]) + compute_DE(base_theta[64:128]) + compute_DE(
                base_theta[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6
            base_alpha_DE = (compute_DE(base_alpha[:64]) + compute_DE(base_alpha[64:128]) + compute_DE(
                base_alpha[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6
            base_beta_DE = (compute_DE(base_beta[:64]) + compute_DE(base_beta[64:128]) + compute_DE(
                base_beta[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6
            base_gamma_DE = (compute_DE(base_gamma[:64]) + compute_DE(base_gamma[64:128]) + compute_DE(
                base_gamma[128:192]) + compute_DE(base_theta[192:256]) + compute_DE(base_theta[256:320]) + compute_DE(
                base_theta[320:])) / 6

            # 添加δ波DE值到临时变量
            temp_base_delta_DE = np.append(temp_base_delta_DE, base_delta_DE)
            temp_base_theta_DE = np.append(temp_base_theta_DE, base_theta_DE)
            temp_base_gamma_DE = np.append(temp_base_gamma_DE, base_gamma_DE)
            temp_base_beta_DE = np.append(temp_base_beta_DE, base_beta_DE)
            temp_base_alpha_DE = np.append(temp_base_alpha_DE, base_alpha_DE)

            # 对测试信号进行δ波滤波
            delta = butter_bandpass_filter(trial_signal, 1, 4, frequency, order=3)
            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 13, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 13, 30, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 30, 50, frequency, order=3)


            # 初始化存储DE值的数组
            DE_delta = np.zeros(shape=[0], dtype=float)
            DE_theta = np.zeros(shape=[0], dtype=float)
            DE_alpha = np.zeros(shape=[0], dtype=float)
            DE_beta = np.zeros(shape=[0], dtype=float)
            DE_gamma = np.zeros(shape=[0], dtype=float)

            for index in range(240):
                DE_delta = np.append(DE_delta, compute_DE(delta[index * 64:(index + 1) * 64]))
                DE_theta = np.append(DE_theta, compute_DE(theta[index * 64:(index + 1) * 64]))
                DE_alpha = np.append(DE_alpha, compute_DE(alpha[index * 64:(index + 1) * 64]))
                DE_beta = np.append(DE_beta, compute_DE(beta[index * 64:(index + 1) * 64]))
                DE_gamma = np.append(DE_gamma, compute_DE(gamma[index * 64:(index + 1) * 64]))

                # 将DE值添加到temp_de中
            temp_de = np.vstack([temp_de, DE_delta])  # 注意添加δ波
            temp_de = np.vstack([temp_de, DE_theta])
            temp_de = np.vstack([temp_de, DE_alpha])
            temp_de = np.vstack([temp_de, DE_beta])
            temp_de = np.vstack([temp_de, DE_gamma])
        # temp_trial_de = temp_de.reshape(-1, 4, 120)
        temp_trial_de = temp_de.reshape(-1, 5, 240)

        decomposed_de = np.vstack([decomposed_de, temp_trial_de])


        # temp_base_DE = np.append(temp_base_theta_DE, temp_base_alpha_DE)
        # temp_base_DE = np.append(temp_base_DE, temp_base_beta_DE)
        # temp_base_DE = np.append(temp_base_DE, temp_base_gamma_DE)
        # base_DE = np.vstack([base_DE, temp_base_DE])

        # 更新基础DE数组
        temp_base_DE = np.append(temp_base_delta_DE, temp_base_theta_DE)  # 注意添加δ波的DE值
        temp_base_DE = np.append(temp_base_DE, temp_base_alpha_DE)  # 添加α波的DE值
        temp_base_DE = np.append(temp_base_DE, temp_base_beta_DE)  # 添加β波的DE值
        temp_base_DE = np.append(temp_base_DE, temp_base_gamma_DE)  # 添加γ波的DE值
        # print('temp_base_DE',temp_base_DE.shape)
        base_DE = np.vstack([base_DE, temp_base_DE])

    # decomposed_de = decomposed_de.reshape(-1, 32, 4, 120).transpose([0, 3, 2, 1]).reshape(-1, 4, 32).reshape(-1, 128)
    # decomposed_de = decomposed_de.reshape(-1, 32, 4, 120).transpose([0, 3, 2, 1]).reshape(-1, 4, 32).transpose(0,2,1)
    decomposed_de = decomposed_de.reshape(-1, 32, 5, 240).transpose([0, 3, 2, 1]).reshape(-1, 5, 32).transpose(0, 2, 1)

    print("base_DE shape:", base_DE.shape)
    print("decomposed_de shape:", decomposed_de.shape)
    return decomposed_de


def get_labels(file):
    # 0 valence, 1 arousal, 2 dominance, 3 liking
    valence_labels = sio.loadmat(file)["labels"][:, 0] > 5  # valence labels
    arousal_labels = sio.loadmat(file)["labels"][:, 1] > 5  # arousal labels
    final_valence_labels = np.empty([0])
    final_arousal_labels = np.empty([0])
    for i in range(len(valence_labels)):
        for j in range(0, 240):
            final_valence_labels = np.append(final_valence_labels, valence_labels[i])
            final_arousal_labels = np.append(final_arousal_labels, arousal_labels[i])
    print("final_arousal_labels:", final_arousal_labels.shape)
    print("final_valence_labels:", final_valence_labels.shape)

    return final_arousal_labels, final_valence_labels


def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data.nonzero()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean) / sigma
    return data_normalized

def save_data_and_labels(decomposed_data, valence_labels, arousal_labels, save_path):
    # 使用np.savez_compressed保存数据和标签到一个文件中
    np.savez_compressed(save_path, data=decomposed_data, valence_labels=valence_labels, arousal_labels=arousal_labels)

if __name__ == '__main__':

    dataset_dir = "J:/参考实验/实验/DEAP/data_preprocessed_matlab/"

    result_dir = "G:/Experiment/Experiment3/Data/DEAP2/"
    os.makedirs(result_dir, exist_ok=True)

    # 假设数据集目录下的文件按照受试者编号排序
    for idx, file in enumerate(sorted(os.listdir(dataset_dir)), 1):
        if file.endswith('.mat'):
            print(f"Processing: {file}...")
            file_path = os.path.join(dataset_dir, file)
            decomposed_data = decompose(file_path)  # 确保decompose函数调整为返回期望的格式（4800, 32, 4）
            valence_labels, arousal_labels = get_labels(file_path)  # 确保get_labels函数返回valence和arousal标签
            save_path = os.path.join(result_dir, f"DE_S{idx}.npz")
            save_data_and_labels(decomposed_data, valence_labels, arousal_labels, save_path)
# import numpy as np
#
# # 加载.npz文件
# data = np.load(r'G:\Experiment\Experiment3\Data\DEAP\DE_S1.npz')
#
# # 打印所有的键值
# print("所有键值：", list(data.keys()))
