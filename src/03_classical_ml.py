# ============================================================
# PHASE 2: Classical Machine Learning
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix,
                             classification_report, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\data"
MODELS_PATH = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\models"
PLOTS_PATH = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\reports"
os.makedirs(MODELS_PATH, exist_ok=True)

# ============================================================
# STEP 1: Load clean data
# ============================================================
print("Loading clean data...")
df = pd.read_parquet(os.path.join(DATA_PATH, 'clean_data.parquet'))
le = joblib.load(os.path.join(DATA_PATH, 'label_encoder.pkl'))
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Separate features and label ───────────────────────────────
X = df.drop('Label', axis=1)
y = df['Label']

print(f"\nFeatures: {X.shape[1]}")
print(f"Classes: {le.classes_}")

# ============================================================
# STEP 2: Train / Test Split
# ============================================================
# We use 80% for training, 20% for testing
# stratify=y ensures each class is proportionally represented
# in both train and test sets
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)
print(f"✅ Train set: {X_train.shape[0]:,} rows")
print(f"✅ Test set:  {X_test.shape[0]:,} rows")

# ============================================================
# STEP 3: Handle Class Imbalance with SMOTE
# ============================================================
print("\nApplying SMOTE to training data...")
print("This may take a few minutes...")

# Check current class counts in training set
unique, counts = np.unique(y_train, return_counts=True)
print("Class counts before SMOTE:")
for u, c in zip(unique, counts):
    print(f"  {le.classes_[u]:30s}: {c:,}")

# Instead of balancing to 2M, we cap at 10,000 per minority class
# This saves memory while still fixing the imbalance problem
sampling_dict = {}
target_count = 10000  # target count for minority classes

for u, c in zip(unique, counts):
    if c < target_count:
        sampling_dict[u] = target_count  # oversample small classes to 10k
    # leave majority classes as they are

print(f"\nSampling strategy: oversample minority classes to {target_count:,}")

smote = SMOTE(
    sampling_strategy=sampling_dict,
    random_state=42,
    k_neighbors=5
)

X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"✅ Before SMOTE: {X_train.shape[0]:,} rows")
print(f"✅ After SMOTE:  {X_train_sm.shape[0]:,} rows")
print("\nClass distribution after SMOTE:")
unique, counts = np.unique(y_train_sm, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {le.classes_[u]:30s}: {c:,}")


# ============================================================
# HELPER FUNCTION: Evaluate any model the same way
# ============================================================
# We write this once and reuse it for all 4 models
def evaluate_model(name, model, X_test, y_test, le):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    
    # Full classification report
    print(f"\n  Per-class report:")
    print(classification_report(y_test, y_pred, 
                                target_names=le.classes_, 
                                zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f'Confusion Matrix — {name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, f'cm_{name.replace(" ", "_")}.png'), dpi=150)
    plt.show()
    print(f"  ✅ Confusion matrix saved")
    
    return {'name': name, 'accuracy': acc, 'precision': prec, 
            'recall': rec, 'f1': f1, 'predictions': y_pred}

# # ============================================================
# # MODEL 1: K-Nearest Neighbors
# # ============================================================
# # You already know KNN from your course!
# # For each new data point, it looks at the K closest 
# # training examples and votes on the label
# # We use a sample for KNN because it's slow on 2M rows
# print("\n" + "="*50)
# print("Training Model 1: KNN")
# print("="*50)

# # KNN is too slow on 2M rows so we train on a sample
# # This is common practice for KNN in production
# print("Sampling 100,000 rows for KNN (it's slow on full data)...")
# sample_idx = np.random.choice(len(X_train_sm), 100000, replace=False)
# X_train_knn = X_train_sm.iloc[sample_idx]
# y_train_knn = y_train_sm.iloc[sample_idx]

# knn = KNeighborsClassifier(
#     n_neighbors=5,      # look at 5 nearest neighbors
#     metric='euclidean', # distance metric
#     n_jobs=-1           # use all CPU cores
# )

# knn.fit(X_train_knn, y_train_knn)
# print("✅ KNN trained!")
# results_knn = evaluate_model("KNN", knn, X_test, y_test, le)
# joblib.dump(knn, os.path.join(MODELS_PATH, 'knn_model.pkl'))
# print("✅ KNN model saved!")

# # ============================================================
# # MODEL 2: Logistic Regression
# # ============================================================
# # Also from your course!
# # Finds a linear boundary to separate classes
# # max_iter=1000 because with 15 classes it needs more iterations
# print("\n" + "="*50)
# print("Training Model 2: Logistic Regression")
# print("="*50)

# lr = LogisticRegression(
#     max_iter=1000,
#     random_state=42,
#     n_jobs=-1,
#     C=1.0  # regularization strength (from your Chapter 7!)
# )

# lr.fit(X_train_sm, y_train_sm)
# print("✅ Logistic Regression trained!")
# results_lr = evaluate_model("Logistic Regression", lr, X_test, y_test, le)
# joblib.dump(lr, os.path.join(MODELS_PATH, 'lr_model.pkl'))
# print("✅ Logistic Regression model saved!")

# # ============================================================
# # MODEL 3: Random Forest
# # ============================================================
# # Random Forest builds many decision trees and combines them
# # Each tree sees a random subset of features and data
# # The final prediction is a vote across all trees
# # This makes it very robust and resistant to overfitting
# print("\n" + "="*50)
# print("Training Model 3: Random Forest")
# print("="*50)

# rf = RandomForestClassifier(
#     n_estimators=100,    # 100 decision trees
#     max_depth=20,        # limit tree depth to avoid overfitting
#     min_samples_split=10,
#     random_state=42,
#     n_jobs=-1,           # use all CPU cores
#     class_weight='balanced'  # extra help for imbalanced classes
# )

# rf.fit(X_train_sm, y_train_sm)
# print("✅ Random Forest trained!")
# results_rf = evaluate_model("Random Forest", rf, X_test, y_test, le)
# joblib.dump(rf, os.path.join(MODELS_PATH, 'rf_model.pkl'))
# print("✅ Random Forest model saved!")

# # ============================================================
# # MODEL 4: XGBoost
# # ============================================================
# # XGBoost = Extreme Gradient Boosting
# # Instead of building trees independently like Random Forest,
# # each tree CORRECTS the mistakes of the previous one
# # This makes it extremely powerful on tabular data
# # It's the most used algorithm in Kaggle competitions
# print("\n" + "="*50)
# print("Training Model 4: XGBoost")
# print("="*50)

# # Get number of classes for XGBoost
# n_classes = len(le.classes_)

# xgb = XGBClassifier(
#     n_estimators=200,        # 200 boosting rounds
#     max_depth=8,             # tree depth
#     learning_rate=0.1,       # how much each tree contributes
#     subsample=0.8,           # use 80% of data per tree
#     colsample_bytree=0.8,    # use 80% of features per tree
#     use_label_encoder=False,
#     eval_metric='mlogloss',  # multi-class log loss
#     random_state=42,
#     n_jobs=-1,
#     tree_method='hist'       # faster training method
# )

# xgb.fit(
#     X_train_sm, y_train_sm,
#     eval_set=[(X_test, y_test)],
#     verbose=50  # print progress every 50 rounds
# )
# print("✅ XGBoost trained!")
# results_xgb = evaluate_model("XGBoost", xgb, X_test, y_test, le)
# joblib.dump(xgb, os.path.join(MODELS_PATH, 'xgb_model.pkl'))
# print("✅ XGBoost model saved!")

# ============================================================
# STEP 5: Model Comparison Chart
# ============================================================
print("\nGenerating model comparison chart...")

# Reload saved results since we ran models separately
results = [
    {'name': 'KNN',                 'accuracy': 0.9853, 'precision': 0.9868, 'recall': 0.9853, 'f1': 0.9860},
    {'name': 'Logistic Regression', 'accuracy': 0.9776, 'precision': 0.9805, 'recall': 0.9776, 'f1': 0.9781},
    {'name': 'Random Forest',       'accuracy': 0.9937, 'precision': 0.9978, 'recall': 0.9937, 'f1': 0.9955},
    {'name': 'XGBoost',             'accuracy': 0.9987, 'precision': 0.9988, 'recall': 0.9987, 'f1': 0.9987},
]

results_df = pd.DataFrame(results)

# ── Bar chart comparison ──────────────────────────────────────
metrics = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(results_df))
width = 0.2
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

fig, ax = plt.subplots(figsize=(14, 7))
for i, (metric, color) in enumerate(zip(metrics, colors)):
    bars = ax.bar(x + i * width, results_df[metric], width, 
                  label=metric.capitalize(), color=color, alpha=0.85)

ax.set_xlabel('Model', fontsize=13)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results_df['name'], fontsize=11)
ax.set_ylim(0.95, 1.005)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, metric in enumerate(metrics):
    for j, val in enumerate(results_df[metric]):
        ax.text(j + i * width, val + 0.0003, f'{val:.4f}', 
                ha='center', va='bottom', fontsize=7, rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, 'model_comparison.png'), dpi=150)
plt.show()
print("✅ Model comparison chart saved!")

# ── Save best model reference ─────────────────────────────────
best_model_name = results_df.loc[results_df['f1'].idxmax(), 'name']
print(f"\n🏆 Best model: {best_model_name}")
print(f"   F1 Score: {results_df.loc[results_df['f1'].idxmax(), 'f1']:.4f}")

# Save results table
results_df.to_csv(os.path.join(PLOTS_PATH, 'model_results.csv'), index=False)
print("✅ Results table saved to reports/model_results.csv")