import numpy as np
import torch
from scipy.signal import find_peaks, welch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.neighbors import KernelDensity
from scipy.stats import skew, kurtosis
import pywt
import networkx as nx
from scipy.stats import entropy
def to_numpy(data):
    """
    将输入数据转换为 NumPy 数组。
    如果数据是 PyTorch 张量，则先转到 CPU 并 detach 后转换为 numpy 数组。
    """
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()
    return data

def mean_fn(data):
    """
    根据数据类型选择使用 NumPy 或 PyTorch 的 mean 函数
    """
    if isinstance(data, torch.Tensor):
        return torch.mean(data)
    else:
        return np.mean(data)

def sum_fn(data):
    """
    根据数据类型选择使用 NumPy 或 PyTorch 的 sum 函数
    """
    if isinstance(data, torch.Tensor):
        return torch.sum(data)
    else:
        return np.sum(data)

def std_fn(data):
    """
    根据数据类型选择使用 NumPy 或 PyTorch 的 std 函数
    """
    if isinstance(data, torch.Tensor):
        return torch.std(data)
    else:
        return np.std(data)

def extract_ecg_features(ecg, fs):
    """提取 ECG 信号特征"""
    ecg = to_numpy(ecg)
    peaks, _ = find_peaks(ecg, distance=fs*0.6)
    if len(peaks) < 2:
        return {"mean_rr": np.nan, "std_rr": np.nan, "rmssd": np.nan, 
                "lf_power": np.nan, "hf_power": np.nan}
    
    rr_intervals = np.diff(peaks) / fs  # 单位：秒
    mean_rr = mean_fn(rr_intervals)
    std_rr = std_fn(rr_intervals)
    rmssd = np.sqrt(mean_fn(np.diff(rr_intervals)**2))
    
    # HRV分析（LF、HF功率）
    f, pxx = welch(rr_intervals, fs=fs, nperseg=256)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    idx_lf = (f >= lf_band[0]) & (f < lf_band[1])
    idx_hf = (f >= hf_band[0]) & (f < hf_band[1])
    lf_power = np.trapz(pxx[idx_lf], f[idx_lf])
    hf_power = np.trapz(pxx[idx_hf], f[idx_hf])

    # 添加非线性特征：样本熵（Sample Entropy）
    sample_entropy = entropy(np.abs(np.diff(rr_intervals)))

    return {"mean_rr": mean_rr, "std_rr": std_rr, "rmssd": rmssd, 
            "lf_power": lf_power, "hf_power": hf_power}

def extract_resp_features(resp, fs):
    """提取呼吸信号特征"""
    resp = to_numpy(resp)
    peaks, _ = find_peaks(resp, distance=fs*0.6)
    if len(peaks) < 2:
        return {"mean_breath_cycle": np.nan, "std_breath_cycle": np.nan, "mean_breath_amplitude": np.nan}
    
    breath_intervals = np.diff(peaks) / fs
    mean_breath_cycle = mean_fn(breath_intervals)
    std_breath_cycle = std_fn(breath_intervals)
    mean_breath_amplitude = mean_fn(resp[peaks])

    # 添加频域特征：呼吸频率和频谱分析
    f, pxx = welch(resp, fs=fs, nperseg=256)
    breathing_freq = f[np.argmax(pxx)]  # 最强频率峰值
    power_breathing_freq = np.max(pxx)  # 对应的功率

    return {"mean_breath_cycle": mean_breath_cycle, "std_breath_cycle": std_breath_cycle, 
            "mean_breath_amplitude": mean_breath_amplitude, "breathing_freq": breathing_freq, 
            "power_breathing_freq": power_breathing_freq}

def extract_emg_features(emg, fs):
    """提取 EMG 信号特征"""
    emg = to_numpy(emg)
    rms = np.sqrt(mean_fn(emg ** 2))
    mav = mean_fn(np.abs(emg))
    iemg = np.sum(np.abs(emg))
    zero_crossings = np.where(np.diff(np.signbit(emg)))[0]
    zcr = len(zero_crossings) / (len(emg) / fs)
    waveform_length = np.sum(np.abs(np.diff(emg)))

    # 频域特征：平均频率
    f, pxx = welch(emg, fs=fs, nperseg=min(256, len(emg)))
    mean_freq = np.sum(f * pxx) / np.sum(pxx) if np.sum(pxx) > 0 else np.nan

    # 添加非线性特征：样本熵（Sample Entropy）
    sample_entropy = entropy(np.abs(np.diff(emg)))

    return {"rms": rms, "mav": mav, "iemg": iemg, "zcr": zcr, 
            "waveform_length": waveform_length, "mean_freq": mean_freq, "sample_entropy": sample_entropy}

def extract_eda_features(eda, fs):
    """提取 EDA 信号特征"""
    eda = to_numpy(eda)
    baseline = mean_fn(eda)
    threshold = baseline + 0.5 * std_fn(eda)
    peaks, properties = find_peaks(eda, height=threshold, distance=fs*1)

    if len(peaks) == 0:
        return {"mean_scl": baseline, "scr_count": 0, "mean_scr_amplitude": 0}
    
    scr_amplitudes = properties["peak_heights"] - baseline
    scr_count = len(peaks)
    mean_scr_amplitude = mean_fn(scr_amplitudes)

    # 添加非线性特征：近似熵（Approximate Entropy）
    approximate_entropy = entropy(np.abs(np.diff(eda)))

    return {"mean_scl": baseline, "scr_count": scr_count, "mean_scr_amplitude": mean_scr_amplitude 
            }

def extract_temp_features(temp, fs):
    """提取 TEMP 信号特征"""
    temp = to_numpy(temp)
    mean_temp = mean_fn(temp)
    std_temp = std_fn(temp)
    t = np.arange(len(temp)) / fs
    # 数据检查
    if len(temp) < 2 or np.var(temp) < 1e-8:
        return {
            'temp_slope': 0.0,
            'temp_mean': np.mean(temp) if len(temp) > 0 else 0.0,
            'temp_std': 0.0 if len(temp) < 2 else np.std(temp)
        }
        
    # 处理可能的NaN/Inf
    temp = np.nan_to_num(temp)
    slope, _ = np.polyfit(t, temp, 1)

    # 添加波动率特征
    diff_temp = np.diff(temp)
    volatility = np.std(diff_temp)

    return {"mean_temp": mean_temp, "std_temp": std_temp, "slope_temp": slope}

def extract_skew_kurtosis_features(signal):
    """
    提取偏度（Skewness）和峰度（Kurtosis）特征
    """
    skewness = skew(signal)
    kurt = kurtosis(signal)
    return {"skewness": skewness, "kurtosis": kurt}

def extract_wavelet_features(signal):
    """
    提取小波变换特征
    """
    coeffs = pywt.wavedec(signal, 'db4', level=4)  # 使用Daubechies 4小波变换
    features = [np.mean(np.abs(c)) for c in coeffs]  # 计算每个小波子带的均值
    return {"wavelet_features": features}

def extract_graph_features(edge_index, num_nodes):
    """
    提取图特征，例如中心性、Pagerank等
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index)
    
    degree_centrality = list(nx.degree_centrality(G).values())
    betweenness_centrality = list(nx.betweenness_centrality(G).values())
    pagerank = list(nx.pagerank(G).values())
    
    return {"degree_centrality": degree_centrality, "betweenness_centrality": betweenness_centrality, "pagerank": pagerank}

def extract_all_features(resp, ecg, emg, eda, temp, fs):
    """
    综合各个信号的特征，将结果合并成一个字典
    """
    features = {}

    # 提取各个信号的特征，并检查是否有返回值
    resp_features = extract_resp_features(resp, fs)
    if resp_features:  # 如果返回值非空，则更新
         features.update({f"resp_{k}": v for k, v in resp_features.items()})

    ecg_features = extract_ecg_features(ecg, fs)
    if ecg_features:
        features.update({f"ecg_{k}": v for k, v in ecg_features.items()})

    emg_features = extract_emg_features(emg, fs)
    if emg_features:
         features.update({f"emg_{k}": v for k, v in emg_features.items()})

    eda_features = extract_eda_features(eda, fs)
    if eda_features:
        features.update({f"eda_{k}": v for k, v in eda_features.items()})

    temp_features = extract_temp_features(temp, fs)
    if temp_features:
        features.update({f"temp_{k}": v for k, v in temp_features.items()})
    
    # 添加偏度和峰度特征
    features.update({"resp_skewness": skew(resp), "resp_kurtosis": kurtosis(resp)})
    features.update({"ecg_skewness": skew(ecg), "ecg_kurtosis": kurtosis(ecg)})
    features.update({"emg_skewness": skew(emg), "emg_kurtosis": kurtosis(emg)})
    features.update({"eda_skewness": skew(eda), "eda_kurtosis": kurtosis(eda)})
    features.update({"temp_skewness": skew(temp), "temp_kurtosis": kurtosis(temp)})

    # # 添加小波变换特征
    # # resp_wavelet_features = extract_wavelet_features(resp)
    # ecg_wavelet_features = extract_wavelet_features(ecg)
    # # emg_wavelet_features = extract_wavelet_features(emg)
    # eda_wavelet_features = extract_wavelet_features(eda)
    # temp_wavelet_features = extract_wavelet_features(temp)

    # # features.update({"resp_wavelet_" + k: v for k, v in resp_wavelet_features.items()})
    # features.update({"ecg_wavelet_" + k: v for k, v in ecg_wavelet_features.items()})
    # # features.update({"emg_wavelet_" + k: v for k, v in emg_wavelet_features.items()})
    # features.update({"eda_wavelet_" + k: v for k, v in eda_wavelet_features.items()})
    # features.update({"temp_wavelet_" + k: v for k, v in temp_wavelet_features.items()})

    return features


def extract_all_features_from_matrix(signal_matrix, fs, mi_threshold=0.1, mi_bandwidth=0.1):
    """
    处理输入为 (5, n_samples) 的生理信号矩阵，各行依次为： 
    [Resp, ECG, EMG, EDA, Temp]
    返回二维特征矩阵，每行代表一个信号的特征
    """
    if signal_matrix.shape[0] != 5:
        raise ValueError("输入矩阵的行数必须为3，分别对应 Resp, ECG, EMG, EDA, Temp")
    
    # 这里假设 signal_matrix 的每一行都是数组或张量
    resp = signal_matrix[0, :]
    ecg = signal_matrix[0, :]
    emg = signal_matrix[2, :]
    eda = signal_matrix[1, :]
    temp = signal_matrix[2, :]

    # 获取特征
    features = extract_all_features( resp, ecg,emg, eda, temp, fs)

    # 检查是否返回 None
    if features is None:
        print("提取特征失败，返回值为 None。")
        return None
     
    # 定义一个辅助函数，用于展开特征值（如果它们是列表、元组或 numpy 数组）
    def flatten_feature(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return list(v)
        else:
            return [v]
        
     # 逐个信号收集特征并展开，确保每个特征都是单个数值
    signals = ['resp','ecg', 'emg', 'eda', 'temp']
    feature_list = []
    for signal in signals:
        flattened_features = []
        # 按键名排序，保证顺序一致
        for k, v in sorted(features.items()):
            if k.startswith(signal):
                flattened_features.extend(flatten_feature(v))
        feature_list.append(flattened_features)
    # 找出所有信号中特征向量的最大长度
    max_length = max(len(vec) for vec in feature_list)

 # 对每个信号的特征向量进行补零，使得长度一致
    padded_feature_list = []
    for vec in feature_list:
        if len(vec) < max_length:
            vec = vec + [0] * (max_length - len(vec))
        padded_feature_list.append(vec)

   

    # 转换为 numpy 数组
    feature_matrix = np.array(padded_feature_list)

    # print("Feature matrix shape:", feature_matrix.shape)  # 打印矩阵的形状，检查是否正确
    # print(feature_matrix)
    
    return feature_matrix


    
def continuous_mutual_info(x, y, bandwidth=0.1):
    """
    基于核密度估计的连续值互信息计算
    Args:
        x, y: 输入的一维数组
        bandwidth: 带宽参数，默认 0.1
    Returns:
        mi: 计算得到的互信息值
    """
    xy = np.vstack((x, y)).T
    kde_xy = KernelDensity(bandwidth=bandwidth).fit(xy)
    kde_x = KernelDensity(bandwidth=bandwidth).fit(x.reshape(-1, 1))
    kde_y = KernelDensity(bandwidth=bandwidth).fit(y.reshape(-1, 1))
    
    log_pxy = kde_xy.score_samples(xy)
    log_px = kde_x.score_samples(x.reshape(-1, 1))
    log_py = kde_y.score_samples(y.reshape(-1, 1))
    
    mi = np.mean(log_pxy - log_px - log_py)
    return mi

def build_edges_from_signals(signals, pearson_thresh=0.1, mi_thresh=0.1, bandwidth=0.1):
    """先使用Pearson快速筛选，再计算互信息"""
    n = len(signals)
    edge_index = []
    edge_attr = []
    
    for i in range(n):
        for j in range(i+1, n):
            # 快速线性检测
            pearson = np.corrcoef(signals[i], signals[j])[0,1]
            if abs(pearson) < pearson_thresh:
                continue
                
            # 通过线性检测后计算互信息
            mi = continuous_mutual_info(signals[i], signals[j])
            if mi > mi_thresh:
                edge_index.extend([[i,j], [j,i]])
                edge_attr.extend([mi, mi])
    
    return edge_index, edge_attr

# class FeatureExtractor(nn.Module):
#     def __init__(self, input_dim, target_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.BatchNorm1d(128),
#             nn.ELU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, target_dim),
#             nn.Tanh()  # 增强非线性表达能力
#         )
    
#     def forward(self, x):
#         return self.net(x)

# class TimeFreqBlock(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         # 时域特征提取
#         self.time_net = nn.Sequential(
#             nn.Conv1d(1, 8, kernel_size=5, padding=2),
#             nn.MaxPool1d(2),
#             nn.ReLU()
#         )
#         # 频域特征提取
#         self.freq_net = nn.Sequential(
#             nn.Linear(input_dim//2, 32),
#             nn.ReLU()
#         )
        
#     def forward(self, x):
#         # 输入x: (batch, features)
#         time_feat = self.time_net(x.unsqueeze(1)).squeeze()
#         freq_feat = self.freq_net(torch.fft.rfft(x, dim=1).abs())
#         return torch.cat([time_feat, freq_feat], dim=1)
    
# class CrossSignalAttention(nn.Module):
#     def __init__(self, target_dim):
#         super().__init__()
#         self.target_dim = target_dim
#         self.query = nn.Linear(target_dim, target_dim)
#         self.key = nn.Linear(target_dim, target_dim)
        
#     def forward(self, node_features):
#         # node_features: (num_signals, target_dim)
#         q = self.query(node_features)
#         k = self.key(node_features)
#         attn = F.softmax(q @ k.T / np.sqrt(self.target_dim), dim=1)
        
#         return attn @ node_features  # 增强的跨信号特征

def build_graph_data(signal_matrix, fs, target_dim=14, mi_threshold=0.1, mi_bandwidth=0.1,device='cpu'):
    """
    构建图数据，包括节点特征矩阵和基于互信息的边信息
    参数:
        signal_matrix: shape 为 (5, n_samples) 的生理信号矩阵，行依次为 Resp, ECG, EMG, EDA, Temp
        fs: 采样率
        target_dim: 每个节点通过全连接层映射后的特征维度
        mi_threshold: 构建边时互信息的阈值
        mi_bandwidth: 互信息计算时的带宽参数
    返回:
        torch_geometric.data.Data 对象，包含节点特征 x, 边索引 edge_index 以及边属性 edge_attr
    """
    # 先提取特征矩阵（每行代表一个信号的特征）
    feature_matrix = extract_all_features_from_matrix(signal_matrix, fs)
    if feature_matrix is None:
        raise ValueError("特征提取失败")
    
    # 对每一行（信号）进行全连接映射，确保所有节点特征维度一致
    nodes = []
    for vec in feature_matrix:
        # 定义一个线性层
        fc = nn.Linear(len(vec), target_dim)
        vec_tensor = torch.tensor(vec, dtype=torch.float)
        mapped = fc(vec_tensor).to(device)
        nodes.append(mapped)
    
    node_features = torch.stack(nodes, dim=0).to(device)
    print("Node feature matrix shape:", node_features.shape)
    
    # 构造基于互信息的边。将原始信号作为输入构造边
    signals = [signal_matrix[i, :].cpu().numpy() if isinstance(signal_matrix, torch.Tensor) 
               else signal_matrix[i, :] for i in range(signal_matrix.shape[0])]
    edge_index_list, edge_attr_list = build_edges_from_signals(signals, pearson_thresh=0.1, mi_thresh=mi_threshold, bandwidth=mi_bandwidth)
    
    if len(edge_index_list) == 0:
        num_nodes = node_features.size(0)
        edge_index_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index_list.append([i, j])
        edge_attr_list = [1.0] * len(edge_index_list)
    
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).to(device)
    
    # 构造 torch_geometric.data.Data 对象
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return data
