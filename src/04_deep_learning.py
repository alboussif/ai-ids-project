# ============================================================
# PHASE 3: Deep Learning
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
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

# ── Reproducibility ───────────────────────────────────────────
# Setting seeds means we get the same results every run
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# STEP 1: Load data
# ============================================================
print("Loading clean data...")
df = pd.read_parquet(os.path.join(DATA_PATH, 'clean_data.parquet'))
le = joblib.load(os.path.join(DATA_PATH, 'label_encoder.pkl'))

X = df.drop('Label', axis=1)
y = df['Label']

n_features = X.shape[1]
n_classes  = len(le.classes_)

print(f"✅ Loaded: {df.shape[0]:,} rows")
print(f"   Features : {n_features}")
print(f"   Classes  : {n_classes}")

# ============================================================
# STEP 2: Train/Test Split + SMOTE
# ============================================================
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Applying SMOTE...")
sampling_dict = {}
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    if c < 10000:
        sampling_dict[u] = 10000

smote = SMOTE(sampling_strategy=sampling_dict, random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"✅ Training set after SMOTE: {X_train_sm.shape[0]:,} rows")

# ── Convert to numpy arrays for Keras ────────────────────────
X_train_np = np.array(X_train_sm)
X_test_np  = np.array(X_test)
y_train_np = np.array(y_train_sm)
y_test_np  = np.array(y_test)

# ── One-hot encode labels for neural network ─────────────────
# Neural networks need labels as vectors not single numbers
# e.g. class 2 (DDoS) becomes [0, 0, 1, 0, 0, 0, ...]
y_train_cat = keras.utils.to_categorical(y_train_np, n_classes)
y_test_cat  = keras.utils.to_categorical(y_test_np,  n_classes)

print(f"\n✅ Data ready for Deep Learning")
print(f"   X_train shape : {X_train_np.shape}")
print(f"   X_test shape  : {X_test_np.shape}")
print(f"   y_train shape : {y_train_cat.shape}")

# ============================================================
# HELPER: Evaluate DL model (same style as classical ML)
# ============================================================
def evaluate_dl_model(name, model, X_test, y_test_cat, y_test_np, le, plots_path):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)  # pick class with highest probability

    acc  = accuracy_score(y_test_np, y_pred)
    prec = precision_score(y_test_np, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test_np, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test_np, y_pred, average='weighted', zero_division=0)

    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    print(f"\n  Per-class report:")
    print(classification_report(y_test_np, y_pred,
                                target_names=le.classes_,
                                zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test_np, y_pred)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f'Confusion Matrix — {name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f'cm_{name.replace(" ", "_")}.png'), dpi=150)
    plt.show()
    print(f"  ✅ Confusion matrix saved")

    return {'name': name, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1}

# ============================================================
# HELPER: Plot training history
# ============================================================
def plot_history(history, name, plots_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train', color='#3498db')
    ax1.plot(history.history['val_accuracy'], label='Validation', color='#e74c3c')
    ax1.set_title(f'{name} — Accuracy', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss plot
    ax2.plot(history.history['loss'], label='Train', color='#3498db')
    ax2.plot(history.history['val_loss'], label='Validation', color='#e74c3c')
    ax2.set_title(f'{name} — Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f'history_{name.replace(" ", "_")}.png'), dpi=150)
    plt.show()
    print(f"  ✅ Training history plot saved")

# # ============================================================
# # MODEL 5: Dense Neural Network
# # ============================================================
# # This is a classic "fully connected" neural network
# # Every neuron in each layer connects to every neuron in the next
# #
# # Architecture:
# # Input(66) → Dense(256) → Dropout → Dense(128) → Dropout
# #           → Dense(64)  → Dropout → Output(15)
# #
# # Dropout: randomly turns off neurons during training
# #          forces the network to not rely on any single neuron
# #          this prevents overfitting (Chapter 7 concept!)
# print("\n" + "="*50)
# print("Training Model 5: Dense Neural Network")
# print("="*50)

# def build_dense_model(n_features, n_classes):
#     model = keras.Sequential([
#         # Input layer
#         layers.Input(shape=(n_features,)),

#         # Hidden layer 1
#         layers.Dense(256, activation='relu'),
#         layers.BatchNormalization(),  # normalizes activations, speeds up training
#         layers.Dropout(0.3),          # randomly drop 30% of neurons

#         # Hidden layer 2
#         layers.Dense(128, activation='relu'),
#         layers.BatchNormalization(),
#         layers.Dropout(0.3),

#         # Hidden layer 3
#         layers.Dense(64, activation='relu'),
#         layers.BatchNormalization(),
#         layers.Dropout(0.2),

#         # Output layer — one neuron per class
#         # softmax converts raw scores to probabilities that sum to 1
#         layers.Dense(n_classes, activation='softmax')
#     ])

#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=0.001),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model

# dense_model = build_dense_model(n_features, n_classes)
# dense_model.summary()

# # Callbacks — these monitor training and stop/adjust automatically
# early_stop = callbacks.EarlyStopping(
#     monitor='val_loss',   # watch validation loss
#     patience=5,           # stop if no improvement for 5 epochs
#     restore_best_weights=True  # use the best weights found
# )

# reduce_lr = callbacks.ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,      # halve the learning rate
#     patience=3,      # after 3 epochs of no improvement
#     min_lr=1e-6
# )

# print("\nTraining Dense Neural Network...")
# dense_history = dense_model.fit(
#     X_train_np, y_train_cat,
#     epochs=30,
#     batch_size=1024,
#     validation_split=0.1,   # use 10% of training data for validation
#     callbacks=[early_stop, reduce_lr],
#     verbose=1
# )

# print("✅ Dense Neural Network trained!")
# results_dense = evaluate_dl_model(
#     "Dense Neural Network",
#     dense_model, X_test_np, y_test_cat, y_test_np, le, PLOTS_PATH
# )
# plot_history(dense_history, "Dense Neural Network", PLOTS_PATH)
# dense_model.save(os.path.join(MODELS_PATH, 'dense_model.keras'))
# print("✅ Dense model saved!")

# ============================================================
# MODEL 6: 1D Convolutional Neural Network (1D-CNN)
# ============================================================
# CNNs were originally designed for images (2D)
# A 1D-CNN treats our 66 features as a sequence
# and slides "filters" across them to detect local patterns
#
# Think of it like this:
# Instead of looking at all 66 features at once,
# it looks at groups of neighboring features together
# This helps detect patterns like:
# "when packet size is high AND flow duration is short → DDoS"
#
# Architecture:
# Input(66,1) → Conv1D → Conv1D → GlobalMaxPool → Dense → Output

print("\n" + "="*50)
print("Training Model 6: 1D-CNN")
print("="*50)

# Reshape data for CNN — needs shape (samples, features, 1)
# Think of it as a 1D "image" with 66 pixels and 1 channel
X_train_cnn = X_train_np.reshape(X_train_np.shape[0], X_train_np.shape[1], 1)
X_test_cnn  = X_test_np.reshape(X_test_np.shape[0], X_test_np.shape[1], 1)
print(f"CNN input shape: {X_train_cnn.shape}")

def build_cnn_model(n_features, n_classes):
    model = keras.Sequential([
        layers.Input(shape=(n_features, 1)),

        # First Conv block
        # 64 filters, each looking at 3 neighboring features
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),  # reduces sequence length by half
        layers.Dropout(0.2),

        # Second Conv block — deeper, more complex patterns
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        # Third Conv block
        layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Global max pooling — takes the strongest signal from each filter
        layers.GlobalMaxPooling1D(),

        # Fully connected head
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        # Output
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

cnn_model = build_cnn_model(n_features, n_classes)
cnn_model.summary()

early_stop_cnn = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr_cnn = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

checkpoint = callbacks.ModelCheckpoint(
    filepath=os.path.join(MODELS_PATH, 'cnn_model_checkpoint.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("\nTraining 1D-CNN...")
cnn_history = cnn_model.fit(
    X_train_cnn, y_train_cat,
    epochs=30,
    batch_size=1024,
    validation_split=0.1,
    callbacks=[early_stop_cnn, reduce_lr_cnn, checkpoint],
    verbose=1
)

print("✅ 1D-CNN trained!")
results_cnn = evaluate_dl_model(
    "1D-CNN",
    cnn_model, X_test_cnn, y_test_cat, y_test_np, le, PLOTS_PATH
)
plot_history(cnn_history, "1D-CNN", PLOTS_PATH)
cnn_model.save(os.path.join(MODELS_PATH, 'cnn_model.keras'))
print("✅ 1D-CNN model saved!")