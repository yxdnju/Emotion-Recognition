import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter, welch


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


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return np.log(2 * np.pi * np.e * variance) / 2


def compute_PSD(signal, fs):
    f, psd = welch(signal, fs, nperseg=1024)
    return psd


def decompose(file, fs):
    data = sio.loadmat(file)['data']
    frequency_bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 50)]

    decomposed_psd = []

    for trial in range(data.shape[0]):
        trial_data = data[trial, :, :]
        trial_psd = []

        for channel in range(trial_data.shape[0]):
            channel_data = trial_data[channel, :]
            channel_psd = []

            for lowcut, highcut in frequency_bands:
                filtered_data = butter_bandpass_filter(channel_data, lowcut, highcut, fs, order=3)
                psd = compute_PSD(filtered_data, fs)
                channel_psd.append(psd)

            trial_psd.append(channel_psd)

        decomposed_psd.append(trial_psd)

    decomposed_psd = np.array(decomposed_psd)
    print("Decomposed data shape:", decomposed_psd.shape)
    return decomposed_psd


def get_labels(file):
    labels = sio.loadmat(file)['labels']
    valence_labels = labels[:, 0] > 5
    arousal_labels = labels[:, 1] > 5
    return valence_labels, arousal_labels


def save_data_and_labels(decomposed_data, valence_labels, arousal_labels, save_path):
    np.savez_compressed(save_path, data=decomposed_data, valence_labels=valence_labels, arousal_labels=arousal_labels)


if __name__ == '__main__':
    dataset_dir = "J:/参考实验/实验/DEAP/data_preprocessed_matlab/"
    result_dir = "G:/Experiment/Experiment3/Data/DEAP_PSD/"
    os.makedirs(result_dir, exist_ok=True)

    fs = 128  # 设定采样频率

    for idx, file in enumerate(sorted(os.listdir(dataset_dir))):
        if file.endswith('.mat'):
            print(f"Processing Subject {idx+1}: {file}...")
            file_path = os.path.join(dataset_dir, file)
            decomposed_data = decompose(file_path, fs)
            valence_labels, arousal_labels = get_labels(file_path)
            save_path = os.path.join(result_dir, f"PSD_{file.split('.')[0]}.npz")
            save_data_and_labels(decomposed_data, valence_labels, arousal_labels, save_path)
