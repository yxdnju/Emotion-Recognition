# FESTFM: Feature-Enhanced Spatiotemporal Fusion Model for Stress Recognition

This repository contains the official implementation of the paper:

> **"Feature-enhanced Multimodal Spatiotemporal Fusion Based on Graph Attention Network for Stress Recognition"**

The proposed **FESTFM** model integrates multimodal physiological signals (ECG, EDA, EMG, RESP, TEMP) for stress recognition using a graph attention network enhanced with feature-level gating, dynamic feature decomposition, and spatiotemporal fusion.

---

## 📊 Datasets

### WESAD Dataset
- **Subjects**: 15
- **Signals**: ECG, EDA, EMG, Temperature, Respiration
- **Classes**: Baseline, Stress, Amusement (3-class)
- **Duration**: ~120 min per subject
- **Sampling Rate**: Downsampled to 250 Hz
- **Access**: [ACM Link](https://dl.acm.org/doi/10.1145/3242969.3242985)

### CASE Dataset
- **Subjects**: 30
- **Signals**: ECG, GSR, EMG, Temperature, Respiration
- **Labels**: Valence (3-class), Arousal (3-class), and 4-class emotion
- **Sampling Rate**: Downsampled to 200 Hz
- **Access**: [Nature Link](https://www.nature.com/articles/s41597-019-0209-0)

---

## ⚙️ Experimental Setup

| Setting | Value |
|---------|-------|
| Time Window | 5 seconds (sliding step 2s) |
| Batch Size | 64 |
| Initial Learning Rate | 3e-3 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | OneCycleLR (pct_start=0.05) |
| Epochs (WESAD) | 100 |
| Epochs (CASE) | 50 |
| Cross-Validation (WESAD) | Leave-One-Subject-Out (LOSO) |
| Cross-Validation (CASE) | 10-fold |
| Framework | PyTorch 1.8.0 + PyTorch Geometric |
| Loss Function | CrossEntropyLoss + 0.1 × PrivateContrastiveLoss |

### Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input Dimension | Depends on extracted features |
| Hidden Dimension | 128 |
| GAT Heads | 4 |
| Feature Attention Dim | 64 |
| Common Feature Dim | 64 (half of hidden) |
| Private Feature Dim (per class) | 32 |
| Time Encoding Dim | 64 |
| Cross-Attention Heads | 4 |
| Dropout | 0.3 - 0.5 |

---

## 📈 Results

### WESAD (3-class)
| Metric | Value |
|--------|-------|
| Accuracy | 97.18% |
| F1-Score | 97.20% |
| Kappa | 0.97 |

### CASE (3-class Valence)
| Metric | Value |
|--------|-------|
| Accuracy | 89.15% |
| F1-Score | 87.63% |

### CASE (3-class Arousal)
| Metric | Value |
|--------|-------|
| Accuracy | 88.72% |
| F1-Score | 87.15% |

### CASE (4-class)
| Metric | Value |
|--------|-------|
| Accuracy | 95.91% |
| F1-Score | 94.91% |

---

## 🧪 Ablation Study Results

| Variant | Accuracy | F1-Score | Kappa |
|---------|----------|----------|-------|
| Full Model | 97.18% | 97.20% | 0.97 |
| w/o FE-GAT (Fully Connected) | 90.36% | 91.31% | 0.83 |
| w/o DFDM | 85.05% | 83.71% | 0.69 |
| w/o Temporal Features | 90.53% | 91.00% | 0.85 |
| w/o STFM | 92.04% | 92.18% | 0.87 |

### Activation Function Comparison
| Activation | Accuracy |
|------------|----------|
| Linear + Sigmoid | 92.11% |
| MLP + ReLU + Sigmoid | 95.94% |
| MLP + GELU + Sigmoid | 96.31% |
| **MLP + Tanh + Sigmoid (Ours)** | **97.18%** |

---

## 📁 Repository Structure


FESTFM/
├── models/
│ └── dynamic_gnn.py # HierarchicalGATConv, FeatureDisentangler, AdaptiveGraphPool, DynamicGNN, TemporalGNN
├── data_processing/
│ ├── dataprocessed_2.py # Data preprocessing and resampling (WESAD)
│ ├── Dataset.py # PyG Dataset wrapper
│ └── extract_features_graph_builder.py # Feature extraction & graph construction
├── training/
│ ├── train_and_detect.py # Training loop with OneCycleLR and early stopping
│ ├── main_1_wesad.py # Main training script for WESAD dataset (LOSO CV)
│ └── main_2_case.py # Main training script for CASE dataset (10-fold CV)
├── losses/
│ └── CustomLoss.py # Focal Loss with L1 regularization
├── utils/
│ └── performance.py # Model performance profiling (FLOPs, parameters, memory)
└── README.md
