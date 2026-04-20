# ============================================================
# PHASE 3: Autoencoder — Anomaly Detection
# ============================================================
# Unlike all previous models, the Autoencoder is UNSUPERVISED
# It only trains on BENIGN traffic and learns what "normal" looks like
# Anything it can't reconstruct well is flagged as an anomaly
# This allows it to detect UNKNOWN attacks never seen before!
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH   = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\data"
MODELS_PATH = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\models"
PLOTS_PATH  = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\reports"

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# STEP 1: Load data
# ============================================================
print("Loading clean data...")
df = pd.read_parquet(os.path.join(DATA_PATH, 'clean_data.parquet'))
le = joblib.load(os.path.join(DATA_PATH, 'label_encoder.pkl'))

X = df.drop('Label', axis=1).values
y = df['Label'].values

print(f"✅ Loaded: {df.shape[0]:,} rows")

# ============================================================
# STEP 2: Prepare data — Autoencoder only trains on BENIGN
# ============================================================
print("\nPreparing data...")

# Get BENIGN label index
benign_idx = np.where(le.classes_ == 'BENIGN')[0][0]
print(f"BENIGN class index: {benign_idx}")

# Separate BENIGN and ATTACK rows
X_benign = X[y == benign_idx]
X_attack = X[y != benign_idx]
y_benign = np.zeros(len(X_benign))  # 0 = normal
y_attack = np.ones(len(X_attack))   # 1 = anomaly

print(f"BENIGN samples : {len(X_benign):,}")
print(f"ATTACK samples : {len(X_attack):,}")

# Split BENIGN into train/test
X_train_ae, X_test_benign = train_test_split(
    X_benign, test_size=0.2, random_state=42
)

# Test set = some benign + all attacks
X_test_ae = np.vstack([X_test_benign, X_attack])
y_test_ae = np.hstack([y_benign[:len(X_test_benign)], y_attack])

print(f"\nAutoencoder training set : {X_train_ae.shape[0]:,} (BENIGN only)")
print(f"Autoencoder test set     : {X_test_ae.shape[0]:,} (BENIGN + ATTACK)")

# ============================================================
# STEP 3: Build the Autoencoder
# ============================================================
# An Autoencoder has two parts:
#
# ENCODER: compresses input from 66 features down to a small
#          "bottleneck" of just 16 numbers (the essence of normal traffic)
#
# DECODER: tries to reconstruct the original 66 features from those 16 numbers
#
# Input(66) → 32 → 16 (bottleneck) → 32 → Output(66)
#
# After training on BENIGN only:
# - BENIGN traffic → reconstructed well → low error
# - ATTACK traffic → reconstructed badly → high error → ANOMALY!

n_features = X.shape[1]

def build_autoencoder(n_features):
    # ── ENCODER ──────────────────────────────────────────────
    encoder = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(16, activation='relu'),  # bottleneck
    ], name='encoder')

    # ── DECODER ──────────────────────────────────────────────
    decoder = keras.Sequential([
        layers.Input(shape=(16,)),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(n_features, activation='linear'),  # reconstruct original
    ], name='decoder')

    # ── FULL AUTOENCODER ─────────────────────────────────────
    inputs = keras.Input(shape=(n_features,))
    encoded = encoder(inputs)
    decoded = decoder(encoded)
    autoencoder = keras.Model(inputs, decoded, name='autoencoder')

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'  # Mean Squared Error — measures reconstruction quality
    )
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder(n_features)
autoencoder.summary()

# # ============================================================
# # STEP 4: Train the Autoencoder
# # ============================================================
# # Note: input = output (X_train_ae, X_train_ae)
# # The model tries to reconstruct its own input!
# print("\nTraining Autoencoder...")

# early_stop_ae = callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     restore_best_weights=True
# )

# reduce_lr_ae = callbacks.ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=3,
#     min_lr=1e-6
# )

# checkpoint_ae = callbacks.ModelCheckpoint(
#     filepath=os.path.join(MODELS_PATH, 'autoencoder_checkpoint.keras'),
#     monitor='val_loss',
#     save_best_only=True,
#     verbose=1
# )

# ae_history = autoencoder.fit(
#     X_train_ae, X_train_ae,
#     epochs=50,
#     batch_size=2048,        # larger batch = more stable gradients
#     validation_split=0.1,
#     shuffle=True,           # shuffle data each epoch
#     callbacks=[early_stop_ae, reduce_lr_ae, checkpoint_ae],
#     verbose=1
# )

# print("✅ Autoencoder trained!")

# # ── Plot training loss ────────────────────────────────────────
# plt.figure(figsize=(10, 5))
# plt.plot(ae_history.history['loss'], label='Train Loss', color='#3498db')
# plt.plot(ae_history.history['val_loss'], label='Val Loss', color='#e74c3c')
# plt.title('Autoencoder Training Loss', fontsize=14, fontweight='bold')
# plt.xlabel('Epoch')
# plt.ylabel('MSE Loss')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(PLOTS_PATH, 'autoencoder_training_loss.png'), dpi=150)
# plt.show()
# print("✅ Training loss plot saved!")

# ============================================================
# STEP 5: Find the anomaly threshold
# ============================================================
# We need to decide: "how high does reconstruction error need
# to be before we call it an anomaly?"
#
# Strategy: train on BENIGN, compute reconstruction errors,
# set threshold at 95th percentile of BENIGN errors
# Anything above this threshold = ANOMALY

print("\nComputing reconstruction errors...")

# Compute errors on training BENIGN data to find threshold
X_train_sample = X_train_ae[:50000]  # sample for speed
train_reconstructed = autoencoder.predict(X_train_sample, verbose=0)
train_errors = np.mean(np.power(X_train_sample - train_reconstructed, 2), axis=1)

# Set threshold at 95th percentile of normal errors
threshold = np.percentile(train_errors, 95)
print(f"✅ Anomaly threshold (95th percentile): {threshold:.6f}")

# Compute errors on test set
print("Computing test set reconstruction errors...")
test_reconstructed = autoencoder.predict(X_test_ae, verbose=0)
test_errors = np.mean(np.power(X_test_ae - test_reconstructed, 2), axis=1)

# Predict: above threshold = anomaly (1), below = normal (0)
y_pred_ae = (test_errors > threshold).astype(int)

# ============================================================
# STEP 6: Evaluate
# ============================================================
print("\n" + "="*50)
print("  Autoencoder — Anomaly Detection")
print("="*50)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc  = accuracy_score(y_test_ae, y_pred_ae)
prec = precision_score(y_test_ae, y_pred_ae, zero_division=0)
rec  = recall_score(y_test_ae, y_pred_ae, zero_division=0)
f1   = f1_score(y_test_ae, y_pred_ae, zero_division=0)

print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1 Score  : {f1:.4f}")

print("\n  Classification Report:")
print(classification_report(y_test_ae, y_pred_ae,
                            target_names=['Normal', 'Anomaly'],
                            zero_division=0))

# ── ROC Curve ────────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test_ae, test_errors)
auc_score = roc_auc_score(y_test_ae, test_errors)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#e74c3c', lw=2,
         label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Autoencoder ROC Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, 'autoencoder_roc.png'), dpi=150)
plt.show()
print(f"✅ ROC curve saved! AUC: {auc_score:.4f}")

# ── Better reconstruction error distribution ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
benign_errors = test_errors[:len(X_test_benign)]
attack_errors = test_errors[len(X_test_benign):]

# Plot 1: Log scale to see both distributions clearly
axes[0].hist(benign_errors, bins=100, alpha=0.7, color='#2ecc71',
             label='Normal Traffic', density=True)
axes[0].hist(attack_errors, bins=100, alpha=0.7, color='#e74c3c',
             label='Attack Traffic', density=True)
axes[0].axvline(threshold, color='black', linestyle='--', lw=2,
                label=f'Threshold = {threshold:.4f}')
axes[0].set_yscale('log')
axes[0].set_xlabel('Reconstruction Error (MSE)', fontsize=12)
axes[0].set_ylabel('Density (log scale)', fontsize=12)
axes[0].set_title('Error Distribution (Log Scale)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Plot 2: Zoom into low error range to see normal traffic clearly
zoom_limit = np.percentile(benign_errors, 99)
axes[1].hist(benign_errors, bins=100, alpha=0.7, color='#2ecc71',
             label=f'Normal (mean={np.mean(benign_errors):.3f})', density=True)
axes[1].hist(attack_errors, bins=100, alpha=0.7, color='#e74c3c',
             label=f'Attack (mean={np.mean(attack_errors):.3f})', density=True)
axes[1].axvline(threshold, color='black', linestyle='--', lw=2,
                label=f'Threshold = {threshold:.4f}')
axes[1].set_xlim(0, zoom_limit)
axes[1].set_xlabel('Reconstruction Error (MSE)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Error Distribution (Zoomed)', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.suptitle('Autoencoder Reconstruction Error Distribution', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, 'autoencoder_error_distribution.png'), 
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Improved error distribution plot saved!")

# ── Save autoencoder and threshold ───────────────────────────
autoencoder.save(os.path.join(MODELS_PATH, 'autoencoder.keras'))
joblib.dump(threshold, os.path.join(MODELS_PATH, 'autoencoder_threshold.pkl'))
print("✅ Autoencoder saved!")
print("✅ Threshold saved!")
print(f"\n🎉 Autoencoder complete!")