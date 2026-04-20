# ============================================================
# PHASE 1 - STEP 2: Preprocessing & Data Cleaning
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import os

# ── Paths ────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\data\MachineLearningCVE"
SAVE_PATH = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\data"

# ── Load data ────────────────────────────────────────────────
print("Loading data...")
all_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
dataframes = []
for file in all_files:
    path = os.path.join(DATA_PATH, file)
    df_temp = pd.read_csv(path, encoding='utf-8', low_memory=False)
    dataframes.append(df_temp)

df = pd.concat(dataframes, ignore_index=True)
df.columns = df.columns.str.strip()
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================
# FIX 1: Replace infinite values with NaN
# ============================================================
print("\nFix 1: Replacing infinite values...")
inf_count_before = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"  Replaced {inf_count_before} infinite values with NaN")

# ============================================================
# FIX 2: Drop rows with missing values
# ============================================================
print("\nFix 2: Dropping rows with missing values...")
rows_before = df.shape[0]
df.dropna(inplace=True)
rows_after = df.shape[0]
print(f"  Dropped {rows_before - rows_after:,} rows")
print(f"  Remaining: {rows_after:,} rows")

# ============================================================
# FIX 3: Drop rows with negative Flow Duration
# ============================================================
print("\nFix 3: Dropping negative Flow Duration rows...")
rows_before = df.shape[0]
df = df[df['Flow Duration'] >= 0]
rows_after = df.shape[0]
print(f"  Dropped {rows_before - rows_after:,} rows")
print(f"  Remaining: {rows_after:,} rows")

# ============================================================
# FIX 4: Drop duplicate rows
# ============================================================
print("\nFix 4: Dropping duplicate rows...")
rows_before = df.shape[0]
df.drop_duplicates(inplace=True)
rows_after = df.shape[0]
print(f"  Dropped {rows_before - rows_after:,} rows")
print(f"  Remaining: {rows_after:,} rows")

# ============================================================
# FIX 5: Fix broken label names
# ============================================================
# ── Extra label fix using regex (catches any encoding variant) ──
import re

print("\nFix 5: Fixing label names...")

def clean_label(label):
    # Replace any weird characters between "Web Attack" and the attack type
    label = re.sub(r'Web Attack\s*.+?\s*(Brute Force|XSS|Sql Injection)', 
                   r'Web Attack - \1', label)
    return label.strip()

df['Label'] = df['Label'].apply(clean_label)
print("\nLabels after regex fix:")
print(df['Label'].value_counts())

# ============================================================
# FIX 6: Remove low variance features
# ============================================================
# Features that barely change across all rows give the model
# no useful information — we drop them
print("\nFix 6: Removing low variance features...")

# Separate features and label
X = df.drop('Label', axis=1)
y = df['Label']

cols_before = X.shape[1]

# VarianceThreshold removes any feature with variance below the threshold
selector = VarianceThreshold(threshold=0.01)
selector.fit(X)

# Get the names of kept columns
kept_cols = X.columns[selector.get_support()].tolist()
dropped_cols = X.columns[~selector.get_support()].tolist()

X = X[kept_cols]

print(f"  Columns before: {cols_before}")
print(f"  Columns after:  {X.shape[1]}")
print(f"  Dropped columns: {dropped_cols}")

# ============================================================
# FIX 7: Encode labels to numbers
# ============================================================
# ML models need numbers not text
# LabelEncoder converts: BENIGN→0, Bot→1, DDoS→2 etc.
print("\nFix 7: Encoding labels...")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"  Label mapping:")
for i, label in enumerate(le.classes_):
    print(f"    {i:2d} → {label}")

# ============================================================
# FIX 8: Scale features
# ============================================================
# Features have very different ranges:
# e.g. Destination Port goes up to 65535
#      Flow Duration goes up to 120,000,000
# StandardScaler brings everything to mean=0, std=1
# This is crucial for models like KNN and Neural Networks
print("\nFix 8: Scaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to dataframe with column names
X_scaled = pd.DataFrame(X_scaled, columns=kept_cols)
print(f"  ✅ Scaled {X_scaled.shape[1]} features")
print(f"  Sample mean (should be ~0): {X_scaled.mean().mean():.6f}")
print(f"  Sample std  (should be ~1): {X_scaled.std().mean():.6f}")

# ============================================================
# SAVE: Store clean data for ML phase
# ============================================================
print("\nSaving cleaned data...")

# Save features
X_scaled['Label'] = y_encoded
X_scaled.to_parquet(os.path.join(SAVE_PATH, 'clean_data.parquet'), index=False)

# Save label encoder classes so we can decode predictions later
import joblib
joblib.dump(le, os.path.join(SAVE_PATH, 'label_encoder.pkl'))
joblib.dump(scaler, os.path.join(SAVE_PATH, 'scaler.pkl'))
joblib.dump(kept_cols, os.path.join(SAVE_PATH, 'feature_columns.pkl'))

print(f"  ✅ Saved clean_data.parquet")
print(f"  ✅ Saved label_encoder.pkl")
print(f"  ✅ Saved scaler.pkl")
print(f"  ✅ Saved feature_columns.pkl")
print(f"\n🎉 Preprocessing complete!")
print(f"   Final dataset: {X_scaled.shape[0]:,} rows × {X_scaled.shape[1]} columns")