# ============================================================
# PHASE 1 - STEP 1: Loading & Exploring the CICIDS2017 Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── 1. Define the path to your data folder ──────────────────
DATA_PATH = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\data\MachineLearningCVE"

# ── 2. Load all CSV files and merge them ────────────────────
print("Loading CSV files...")

all_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
print(f"Found {len(all_files)} files:")
for f in all_files:
    print(f"  - {f}")

# Load each file into a list, then concatenate into one dataframe
dataframes = []
for file in all_files:
    path = os.path.join(DATA_PATH, file)
    df_temp = pd.read_csv(path, encoding='utf-8', low_memory=False)
    print(f"  Loaded {file}: {df_temp.shape[0]:,} rows")
    dataframes.append(df_temp)

# Merge all into one big dataframe
df = pd.concat(dataframes, ignore_index=True)
print(f"\n✅ Total dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================
# STEP 2: Deep Inspection
# ============================================================

# ── 1. Fix column names (strip leading/trailing spaces) ──────
df.columns = df.columns.str.strip()
print("✅ Column names cleaned")
print(df.columns.tolist())

# ── 2. Check label distribution ──────────────────────────────
print("\n--- Attack Label Distribution ---")
label_counts = df['Label'].value_counts()
print(label_counts)
print(f"\nTotal unique labels: {df['Label'].nunique()}")

# ── 3. Check for missing values ──────────────────────────────
print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("✅ No missing values found")
else:
    print(missing)

# ── 4. Check for infinite values ─────────────────────────────
print("\n--- Infinite Values ---")
inf_counts = np.isinf(df.select_dtypes(include=np.number)).sum()
inf_counts = inf_counts[inf_counts > 0]
if len(inf_counts) == 0:
    print("✅ No infinite values found")
else:
    print(inf_counts)

# ── 5. Check for negative flow duration ──────────────────────
print("\n--- Negative Flow Duration ---")
neg_duration = df[df['Flow Duration'] < 0].shape[0]
print(f"Rows with negative Flow Duration: {neg_duration}")

# ── 3. First look at the data ────────────────────────────────

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Column names ---")
print(df.columns.tolist())

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Basic statistics ---")
print(df.describe())

# ============================================================
# STEP 3: Visualisation
# ============================================================

# Create a folder to save plots
PLOTS_PATH = r"C:\Users\hp\Desktop\Projets\cyber-ids-project\reports"
os.makedirs(PLOTS_PATH, exist_ok=True)

# ── 1. Attack Label Distribution Bar Chart ───────────────────
plt.figure(figsize=(14, 6))
colors = ['#2ecc71' if label == 'BENIGN' else '#e74c3c' for label in label_counts.index]
bars = plt.bar(label_counts.index, label_counts.values, color=colors, edgecolor='black', linewidth=0.5)

# Add value labels on top of each bar
for bar, val in zip(bars, label_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
             f'{val:,}', ha='center', va='bottom', fontsize=8, rotation=45)

plt.title('Traffic Label Distribution in CICIDS2017', fontsize=16, fontweight='bold')
plt.xlabel('Traffic Type', fontsize=12)
plt.ylabel('Number of Flows', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, '01_label_distribution.png'), dpi=150)
plt.show()
print("✅ Saved label distribution plot")

# ── 2. Pie Chart — Attack vs Benign ──────────────────────────
plt.figure(figsize=(8, 8))
attack_count = df[df['Label'] != 'BENIGN'].shape[0]
benign_count = df[df['Label'] == 'BENIGN'].shape[0]

plt.pie(
    [benign_count, attack_count],
    labels=['BENIGN', 'ATTACK'],
    colors=['#2ecc71', '#e74c3c'],
    autopct='%1.2f%%',
    startangle=90,
    explode=(0, 0.05),
    shadow=True,
    textprops={'fontsize': 13}
)
plt.title('Benign vs Attack Traffic', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, '02_benign_vs_attack.png'), dpi=150)
plt.show()
print("✅ Saved benign vs attack pie chart")

# ── 3. Log Scale Distribution (to see rare attacks clearly) ──
plt.figure(figsize=(14, 6))
plt.bar(label_counts.index, label_counts.values, color=colors, edgecolor='black', linewidth=0.5)
plt.yscale('log')
plt.title('Traffic Label Distribution (Log Scale)', fontsize=16, fontweight='bold')
plt.xlabel('Traffic Type', fontsize=12)
plt.ylabel('Number of Flows (log scale)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, '03_label_distribution_log.png'), dpi=150)
plt.show()
print("✅ Saved log scale distribution plot")

# ── 4. Missing & Infinite Values Heatmap ─────────────────────
# Create a summary dataframe of data quality issues
issue_df = pd.DataFrame({
    'Missing': df.isnull().sum(),
    'Infinite': np.isinf(df.select_dtypes(include=np.number)).sum()
})
issue_df = issue_df[issue_df.sum(axis=1) > 0]  # only show columns with issues

plt.figure(figsize=(8, 4))
sns.heatmap(issue_df.T, annot=True, fmt='.0f', cmap='Reds', linewidths=0.5)
plt.title('Data Quality Issues per Column', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, '04_data_quality.png'), dpi=150)
plt.show()
print("✅ Saved data quality heatmap")

print("\n✅ All plots saved to reports folder!")