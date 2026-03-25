import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, time, json

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense,
    Dropout, BatchNormalization, Multiply, Permute,
    Flatten, RepeatVector, Lambda, Activation
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

np.random.seed(42)
tf.random.set_seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  Edge-Enhanced CNN-LSTM IDS — Model Training")
print("=" * 60)


# 1. Load Data

print("\n[1] Loading preprocessed datasets ...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
label_map = pd.read_csv(os.path.join(DATA_DIR, "label_map.csv"))

feature_cols = [c for c in train_df.columns if c != "label"]
NUM_CLASSES  = len(label_map)

X_train = train_df[feature_cols].values.astype(np.float32)
y_train = train_df["label"].values
X_val   = val_df[feature_cols].values.astype(np.float32)
y_val   = val_df["label"].values

# Reshape for Conv1D: (samples, timesteps, features) — treat each feature as a timestep
X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_val   = X_val.reshape(-1, X_val.shape[1], 1)

Y_train = to_categorical(y_train, NUM_CLASSES)
Y_val   = to_categorical(y_val,   NUM_CLASSES)

print(f"    X_train shape : {X_train.shape}")
print(f"    X_val shape   : {X_val.shape}")
print(f"    Num classes   : {NUM_CLASSES}")


# 2. Attention Mechanism

def attention_block(inputs):
    """Soft attention over LSTM output timesteps."""
    # inputs shape: (batch, timesteps, features)
    score = Dense(1, activation="tanh")(inputs)          # (batch, timesteps, 1)
    score = Flatten()(score)                              # (batch, timesteps)
    score = Activation("softmax")(score)                  # (batch, timesteps)
    score = RepeatVector(inputs.shape[-1])(score)         # (batch, features, timesteps)
    score = Permute([2, 1])(score)                        # (batch, timesteps, features)
    weighted = Multiply()([inputs, score])                # (batch, timesteps, features)
    # Sum over timesteps
    context = Lambda(lambda x: K.sum(x, axis=1))(weighted)  # (batch, features)
    return context


# 3. Build Model

print("\n[2] Building CNN-LSTM + Attention model ...")

inp = Input(shape=(X_train.shape[1], 1), name="network_flow_input")

# --- CNN Block ---
x = Conv1D(filters=64, kernel_size=5, padding="same",
           activation="relu", name="conv1")(inp)
x = BatchNormalization(name="bn1")(x)
x = MaxPooling1D(pool_size=2, name="pool1")(x)
x = Dropout(0.2, name="drop1")(x)

x = Conv1D(filters=128, kernel_size=3, padding="same",
           activation="relu", name="conv2")(x)
x = BatchNormalization(name="bn2")(x)
x = MaxPooling1D(pool_size=2, name="pool2")(x)
x = Dropout(0.2, name="drop2")(x)

# --- LSTM Block ---
x = LSTM(128, return_sequences=True, name="lstm1")(x)
x = Dropout(0.3, name="drop3")(x)
x = LSTM(64,  return_sequences=True, name="lstm2")(x)

# --- Attention Block ---
x = attention_block(x)

# --- Classifier ---
x = Dense(128, activation="relu", name="fc1")(x)
x = Dropout(0.3, name="drop4")(x)
x = Dense(64,  activation="relu", name="fc2")(x)
out = Dense(NUM_CLASSES, activation="softmax", name="output")(x)

model = Model(inputs=inp, outputs=out, name="Edge_CNN_LSTM_Attention_IDS")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Count params
total_params = model.count_params()
print(f"\n    Total trainable parameters: {total_params:,}")


# 4. Callbacks

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ModelCheckpoint(os.path.join(OUT_DIR, "best_model.keras"),
                    save_best_only=True, monitor="val_accuracy", verbose=0),
    CSVLogger(os.path.join(OUT_DIR, "training_log.csv"))
]


# 5. Train

print("\n[3] Starting training ...")
EPOCHS     = 30
BATCH_SIZE = 256

t0 = time.time()
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)
elapsed = time.time() - t0

best_epoch  = int(np.argmax(history.history["val_accuracy"])) + 1
best_val_acc = max(history.history["val_accuracy"])
best_val_loss = min(history.history["val_loss"])

print(f"\n    Training time : {elapsed:.1f}s")
print(f"    Best epoch    : {best_epoch}")
print(f"    Best val acc  : {best_val_acc:.4f}")
print(f"    Best val loss : {best_val_loss:.4f}")


# 6. Plot Training Curves

print("\n[4] Saving training curves ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(history.history["accuracy"],     label="Train Accuracy", color="#2196F3", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy",   color="#FF5722", linewidth=2, linestyle="--")
axes[0].axvline(best_epoch - 1, color="green", linestyle=":", linewidth=1.5, label=f"Best epoch ({best_epoch})")
axes[0].set_title("Model Accuracy over Epochs", fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["loss"],     label="Train Loss", color="#2196F3", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Val Loss",   color="#FF5722", linewidth=2, linestyle="--")
axes[1].axvline(best_epoch - 1, color="green", linestyle=":", linewidth=1.5, label=f"Best epoch ({best_epoch})")
axes[1].set_title("Model Loss over Epochs", fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle("Edge-Enhanced CNN-LSTM Attention IDS — Training Curves",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved → outputs/training_curves.png")

# 7. Save model summary and training meta

meta = {
    "model_name"       : "Edge_CNN_LSTM_Attention_IDS",
    "total_params"     : total_params,
    "num_classes"      : NUM_CLASSES,
    "epochs_run"       : len(history.history["accuracy"]),
    "best_epoch"       : best_epoch,
    "best_val_accuracy": round(best_val_acc, 6),
    "best_val_loss"    : round(best_val_loss, 6),
    "training_time_s"  : round(elapsed, 2),
    "batch_size"       : BATCH_SIZE,
    "optimizer"        : "Adam(lr=1e-3)",
    "loss"             : "categorical_crossentropy"
}
with open(os.path.join(OUT_DIR, "training_meta.json"), "w") as f:
    json.dump(meta, f, indent=4)
print(f"    Saved → outputs/training_meta.json")
print(f"    Saved → outputs/training_log.csv")
print(f"    Saved → outputs/best_model.keras")

print(f"\n{'=' * 60}")
print(f"  Training complete.")
print(f"{'=' * 60}\n")
