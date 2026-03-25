import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = sns.color_palette("tab10", 12)

print("=" * 60)
print("  CICIDS2017 — Exploratory Data Analysis")
print("=" * 60)

# Load
print("\n[1] Loading preprocessed training data ...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
label_map = pd.read_csv(os.path.join(DATA_DIR, "label_map.csv"))
lm = dict(zip(label_map["encoded"], label_map["label"]))
train["class_name"] = train["label"].map(lm)
print(f"    Loaded {len(train):,} training samples, {train.shape[1]} columns")

# 1. Class distribution bar chart
print("\n[2] Plotting class distribution ...")
class_counts = train["class_name"].value_counts()
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(class_counts.index, class_counts.values, color=COLORS, edgecolor="white", linewidth=0.7)
for bar, val in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"{val:,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_title("Class Distribution — CICIDS2017 Training Set", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Attack / Traffic Class", fontsize=11)
ax.set_ylabel("Number of Samples", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.xticks(rotation=35, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=150)
plt.close()
print(f"    Saved → outputs/class_distribution.png")

# 2. Feature correlation heatmap (top 20 features)
print("\n[3] Plotting feature correlation heatmap (top 20 features) ...")
top_features = train.drop(columns=["label", "class_name"]).var().nlargest(20).index.tolist()
corr = train[top_features].corr()
fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", linewidths=0.4,
            vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap — Top 20 High-Variance Features",
             fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()
print(f"    Saved → outputs/correlation_heatmap.png")

# 3. Flow Duration distribution by class
print("\n[4] Plotting Flow Duration distribution by class ...")
fig, ax = plt.subplots(figsize=(12, 5))
for i, (cls, grp) in enumerate(train.groupby("class_name")["Flow Duration"]):
    ax.hist(grp, bins=50, alpha=0.55, color=COLORS[i % 12], label=cls, density=True)
ax.set_title("Flow Duration Distribution by Traffic Class", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Flow Duration (normalized)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.legend(fontsize=7.5, ncol=3, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "flow_duration_distribution.png"), dpi=150)
plt.close()
print(f"    Saved → outputs/flow_duration_distribution.png")

# 4. Packet length boxplot
print("\n[5] Plotting packet length statistics per class ...")
fig, ax = plt.subplots(figsize=(13, 6))
data_list = [train[train["class_name"] == c]["Packet Length Mean"].values for c in class_counts.index]
bp = ax.boxplot(data_list, patch_artist=True, notch=False, vert=True,
                medianprops=dict(color="black", linewidth=1.8))
for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_xticklabels(class_counts.index, rotation=35, ha="right", fontsize=9)
ax.set_title("Packet Length Mean — Distribution per Traffic Class",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Traffic Class", fontsize=11)
ax.set_ylabel("Packet Length Mean (normalized)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "packet_length_boxplot.png"), dpi=150)
plt.close()
print(f"    Saved → outputs/packet_length_boxplot.png")

# 5. Summary stats
print("\n[6] Computing summary statistics ...")
numeric_cols = train.drop(columns=["label", "class_name"])
summary = numeric_cols.describe().T[["mean", "std", "min", "max"]]
summary.to_csv(os.path.join(OUT_DIR, "feature_summary_stats.csv"))
print(f"    Saved → outputs/feature_summary_stats.csv")

print(f"\n{'=' * 60}")
print(f"  EDA complete. All outputs saved to eda/outputs/")
print(f"{'=' * 60}\n")
