# ============================================================
# PHASE 4: SHAP Explainability
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH   = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\data"
MODELS_PATH = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\models"
PLOTS_PATH  = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\reports"

# ============================================================
# STEP 1: Load data and model
# ============================================================
print("Loading data and model...")
df = pd.read_parquet(os.path.join(DATA_PATH, 'clean_data.parquet'))
le = joblib.load(os.path.join(DATA_PATH, 'label_encoder.pkl'))
xgb = joblib.load(os.path.join(MODELS_PATH, 'xgb_model.pkl'))

X = df.drop('Label', axis=1)
y = df['Label']

feature_names = X.columns.tolist()

# Use a sample for SHAP — computing on 2M rows would take forever
# 5000 rows is more than enough for reliable SHAP values
print("Sampling 5000 rows for SHAP analysis...")
sample_idx = np.random.choice(len(X), 5000, replace=False)
X_sample = X.iloc[sample_idx]
y_sample = y.iloc[sample_idx]

print(f"✅ Sample shape: {X_sample.shape}")
print(f"✅ Model loaded: XGBoost")

# ============================================================
# STEP 2: Compute SHAP values
# ============================================================
# TreeExplainer is optimized for tree-based models like XGBoost
# It's much faster than the generic explainer
print("\nComputing SHAP values (this takes a few minutes)...")

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_sample)

print(f"✅ SHAP values computed!")
print(f"   Shape: {np.array(shap_values).shape}")
# Shape is (n_classes, n_samples, n_features)
# One set of SHAP values per class

# ============================================================
# STEP 3: Global Feature Importance
# ============================================================
# Mean absolute SHAP value across all classes and samples
# Shows which features matter most overall
print("\nGenerating global feature importance plot...")

# Average SHAP importance across all classes
shap_importance = np.mean(np.abs(shap_values), axis=2)  # avg across classes (axis=2)
mean_importance = np.mean(shap_importance, axis=0)       # avg across samples (axis=0)

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': mean_importance
}).sort_values('importance', ascending=False)

print("\nTop 15 most important features:")
print(importance_df.head(15).to_string(index=False))

# Plot top 20 features
plt.figure(figsize=(12, 8))
top20 = importance_df.head(20)
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 20))
bars = plt.barh(range(20), top20['importance'].values, color=colors)
plt.yticks(range(20), top20['feature'].values, fontsize=10)
plt.xlabel('Mean |SHAP Value|', fontsize=12)
plt.title('Top 20 Most Important Features (Global)', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, 'shap_global_importance.png'), dpi=150)
plt.show()
print("✅ Global importance plot saved!")

# ============================================================
# STEP 4: Per-class SHAP analysis
# ============================================================
# Show which features matter most for detecting each attack type
print("\nGenerating per-class SHAP plots...")

# Pick the most interesting attack classes to visualize
interesting_classes = {
    2: 'DDoS',
    4: 'DoS Hulk', 
    7: 'FTP-Patator',
    10: 'PortScan',
    12: 'Web Attack - Brute Force'
}

fig, axes = plt.subplots(1, len(interesting_classes), figsize=(20, 6))

for idx, (class_idx, class_name) in enumerate(interesting_classes.items()):
    # Get SHAP values for this class
    class_shap = shap_values[:, :, class_idx]  # shape: (5000, 66)
    
    # Get top 10 features for this class
    mean_abs = np.mean(np.abs(class_shap), axis=0)
    top10_idx = np.argsort(mean_abs)[-10:][::-1]
    top10_features = [feature_names[i] for i in top10_idx]
    top10_values = mean_abs[top10_idx]
    
    # Plot
    axes[idx].barh(range(10), top10_values, 
                   color=plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, 10)))
    axes[idx].set_yticks(range(10))
    axes[idx].set_yticklabels(top10_features, fontsize=7)
    axes[idx].set_title(class_name, fontsize=10, fontweight='bold')
    axes[idx].invert_yaxis()
    axes[idx].grid(axis='x', alpha=0.3)

plt.suptitle('Top 10 Features per Attack Type (SHAP)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, 'shap_per_class.png'), dpi=150)
plt.show()
print("✅ Per-class SHAP plot saved!")

# ============================================================
# STEP 5: Single prediction explanation
# ============================================================
# This is the most impressive part — explaining ONE prediction
# "Why did the model flag THIS specific flow as DDoS?"
print("\nGenerating single prediction explanation...")

# Find a DDoS sample in our test data
ddos_label = le.transform(['DDoS'])[0]
ddos_mask = y_sample.values == ddos_label
ddos_samples = X_sample[ddos_mask]

if len(ddos_samples) > 0:
    # Take the first DDoS sample
    single_sample = ddos_samples.iloc[0:1]
    single_shap = shap_values[np.where(ddos_mask)[0][0], :, ddos_label]
    
    # Get top 10 contributing features for this prediction
    top_idx = np.argsort(np.abs(single_shap))[-10:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_shap_vals = single_shap[top_idx]
    
    # Plot waterfall-style
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in top_shap_vals]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(10), top_shap_vals, color=colors, edgecolor='black', lw=0.5)
    plt.yticks(range(10), top_features, fontsize=10)
    plt.axvline(0, color='black', lw=1)
    plt.xlabel('SHAP Value (red = towards attack, green = towards normal)', fontsize=10)
    plt.title('Single Prediction Explanation — DDoS Detection', 
              fontsize=13, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'shap_single_prediction.png'), dpi=150)
    plt.show()
    print("✅ Single prediction explanation saved!")
else:
    print("No DDoS samples in this random sample, try re-running")

# ============================================================
# STEP 6: Save SHAP values for web app
# ============================================================
joblib.dump(shap_values, os.path.join(MODELS_PATH, 'shap_values.pkl'))
joblib.dump(importance_df, os.path.join(MODELS_PATH, 'shap_importance.pkl'))
print("\n✅ SHAP values saved!")
print("\n🎉 SHAP explainability complete!")