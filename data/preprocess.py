"""
preprocess.py
-------------
Simulates loading the CICIDS2017 dataset, cleans it, encodes labels,
normalizes features, and saves the processed splits to CSV files.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

np.random.seed(42)

ATTACK_TYPES = [
    "BENIGN", "DoS Hulk", "PortScan", "DDoS",
    "DoS GoldenEye", "FTP-Patator", "SSH-Patator",
    "DoS slowloris", "DoS Slowhttptest", "Bot",
    "Web Attack – Brute Force", "Infiltration"
]

LABEL_DIST = [0.45, 0.15, 0.10, 0.08, 0.05, 0.04,
              0.03, 0.03, 0.02, 0.02, 0.02, 0.01]

N_SAMPLES = 18000
N_FEATURES = 78

FEATURE_NAMES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
    "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
    "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

print("=" * 60)
print("  CICIDS2017 — Data Preprocessing Pipeline")
print("=" * 60)
print(f"\n[1] Generating synthetic CICIDS2017-style dataset ...")
print(f"    Samples : {N_SAMPLES:,}")
print(f"    Features: {N_FEATURES}")
print(f"    Classes : {len(ATTACK_TYPES)}")

counts = (np.array(LABEL_DIST) * N_SAMPLES).astype(int)
counts[-1] += N_SAMPLES - counts.sum()

labels = np.repeat(ATTACK_TYPES, counts)
np.random.shuffle(labels)

X = np.random.exponential(scale=2.0, size=(N_SAMPLES, N_FEATURES))
X = np.clip(X, 0, 1e6)

df = pd.DataFrame(X, columns=FEATURE_NAMES)
df[" Label"] = labels

print(f"\n[2] Class distribution:")
for cls, cnt in df[" Label"].value_counts().items():
    pct = cnt / N_SAMPLES * 100
    print(f"    {cls:<35} {cnt:>5}  ({pct:.1f}%)")

print(f"\n[3] Cleaning: dropping inf/NaN values ...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
before = len(df)
df.dropna(inplace=True)
print(f"    Rows before: {before:,}  |  After: {len(df):,}")

print(f"\n[4] Encoding labels ...")
le = LabelEncoder()
df["Label_Encoded"] = le.fit_transform(df[" Label"])
print(f"    Classes encoded: {list(le.classes_)}")

print(f"\n[5] Normalizing features with MinMaxScaler ...")
scaler = MinMaxScaler()
feature_cols = FEATURE_NAMES
df[feature_cols] = scaler.fit_transform(df[feature_cols])
print(f"    Feature range after scaling: [0.0, 1.0]")

print(f"\n[6] Splitting into Train / Validation / Test sets ...")
X_data = df[feature_cols].values
y_data = df["Label_Encoded"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X_data, y_data, test_size=0.30, random_state=42, stratify=y_data
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"    Train : {len(X_train):,} samples (70%)")
print(f"    Val   : {len(X_val):,} samples (15%)")
print(f"    Test  : {len(X_test):,} samples (15%)")

os.makedirs("outputs", exist_ok=True)

pd.DataFrame(X_train, columns=feature_cols).assign(label=y_train).to_csv(
    "outputs/train.csv", index=False
)
pd.DataFrame(X_val, columns=feature_cols).assign(label=y_val).to_csv(
    "outputs/val.csv", index=False
)
pd.DataFrame(X_test, columns=feature_cols).assign(label=y_test).to_csv(
    "outputs/test.csv", index=False
)

label_map = dict(zip(range(len(le.classes_)), le.classes_))
pd.DataFrame(list(label_map.items()), columns=["encoded", "label"]).to_csv(
    "outputs/label_map.csv", index=False
)

print(f"\n[7] Saved outputs:")
print(f"    outputs/train.csv   ({len(X_train):,} rows)")
print(f"    outputs/val.csv     ({len(X_val):,} rows)")
print(f"    outputs/test.csv    ({len(X_test):,} rows)")
print(f"    outputs/label_map.csv")
print(f"\n{'=' * 60}")
print(f"  Preprocessing complete.")
print(f"{'=' * 60}\n")
