# Edge-Enhanced CNN-LSTM Intrusion Detection System
### Master's Thesis — Advanced Computing | Morgan State University

> **IoT Network Intrusion Detection using a Hybrid CNN-LSTM Architecture with Attention Mechanism, optimised for Edge Deployment on Resource-Constrained Devices**

---

## Repository Structure

```
thesis-iot-ids/
├── data/
│   ├── preprocess.py              # Data loading, cleaning, normalisation, splitting
│   └── outputs/
│       ├── train.csv              # 12,600 training samples
│       ├── val.csv                # 2,700 validation samples
│       ├── test.csv               # 2,700 test samples
│       ├── label_map.csv          # Encoded label → class name
│       └── preprocess_log.txt     # Full preprocessing log
│
├── eda/
│   ├── exploratory_analysis.py    # EDA: class distribution, correlations, plots
│   └── outputs/
│       ├── class_distribution.png
│       ├── correlation_heatmap.png
│       ├── flow_duration_distribution.png
│       ├── packet_length_boxplot.png
│       └── feature_summary_stats.csv
│
├── model/
│   ├── train_model.py             # CNN-LSTM + Attention model definition & training
│   └── outputs/
│       ├── best_model.keras       # Saved model weights (best val accuracy)
│       ├── training_curves.png    # Accuracy & loss per epoch
│       ├── training_log.csv       # Epoch-by-epoch metrics
│       └── training_meta.json     # Hyperparameters & training summary
│
├── evaluation/
│   ├── evaluate.py                # Metrics, confusion matrix, ROC curves
│   └── outputs/
│       ├── classification_report.txt
│       ├── confusion_matrix.png
│       ├── roc_curves.png
│       └── evaluation_summary.json
│
├── edge/
│   ├── edge_deployment.py         # TFLite conversion & latency benchmarking
│   └── outputs/
│       ├── model_float32.tflite   # TFLite float32
│       ├── model_int8.tflite      # TFLite INT8 quantized (69.8% smaller)
│       ├── edge_comparison.csv
│       ├── edge_comparison.png
│       └── edge_summary.json
│
├── results/
│   └── final_summary.txt          # Complete results summary
│
└── README.md
```

---

## Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset  
- 78 flow-based features extracted using CICFlowMeter  
- 12 traffic classes: BENIGN + 11 attack types (DoS, DDoS, PortScan, Brute Force, Bot, Infiltration, Web Attack)  
- Download: https://www.unb.ca/cic/datasets/ids-2017.html

---

## Model Architecture

```
Input (78 features)
     │
  Conv1D(64, k=5) → BatchNorm → MaxPool → Dropout(0.2)
     │
  Conv1D(128, k=3) → BatchNorm → MaxPool → Dropout(0.2)
     │
  LSTM(128, return_sequences=True) → Dropout(0.3)
     │
  LSTM(64, return_sequences=True)
     │
  Soft Attention (Dense → Softmax → Weighted Sum)
     │
  Dense(128, relu) → Dropout(0.3) → Dense(64, relu)
     │
  Dense(12, softmax)
```

Total trainable parameters: **224,269**

---

## Results

| Metric             | Value   |
|--------------------|---------|
| Overall Accuracy   | 98.85%  |
| Macro Precision    | 96.88%  |
| Macro Recall       | 98.06%  |
| Macro F1-Score     | 97.45%  |
| Mean AUC (OVR)     | 98.79%  |

### Edge Deployment

| Format          | Size (KB) | Latency (ms/sample) | Accuracy Drop |
|-----------------|-----------|---------------------|---------------|
| Keras float32   | 875.8     | 2.519               | —             |
| TFLite float32  | 897.3     | 1.774               | 0.00%         |
| TFLite INT8     | 271.2     | 1.172               | ~0.18%        |

INT8 quantization achieves **69.8% model size reduction** and **2.15× speedup** over the full Keras model, making it suitable for deployment on Raspberry Pi, NVIDIA Jetson Nano, and similar IoT gateway devices.

---

## How to Run

Each folder contains a standalone script. Run them in order:

```bash
# 1. Preprocess data
cd data && python preprocess.py

# 2. Exploratory Data Analysis
cd eda && python exploratory_analysis.py

# 3. Train the model
cd model && python train_model.py

# 4. Evaluate on test set
cd evaluation && python evaluate.py

# 5. Edge deployment & TFLite conversion
cd edge && python edge_deployment.py
```

### Requirements

```
tensorflow >= 2.13
numpy
pandas
scikit-learn
matplotlib
seaborn
```

Install with:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

---

## References

- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *ICISSP*, 108–116.
- Vinayakumar, R., Alazab, M., Soman, K. P., et al. (2019). Deep learning approach for intelligent intrusion detection system. *IEEE Access*, 7, 41525–41550.
- Kim, J., Kim, J., Kim, H., Shim, M., & Choi, E. (2019). CNN-LSTM based anomaly detection for time-series IoT data. *Electronics*, 8(11), 1362.
