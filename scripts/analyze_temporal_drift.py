"""
Analyze feature drift between train and test to identify 30% Gini gap cause

This script investigates why model performance drops from 0.63 to 0.42
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TEMPORAL DRIFT ANALYSIS - Finding the 30% Gini Gap Cause")
print("=" * 80)

# Load temporal splits
print("\n[1/5] Loading temporal train/test splits...")
train = pd.read_csv('Data/data_train_temporal.csv')
test = pd.read_csv('Data/data_test_temporal.csv')

print(f"Train: {len(train):,} samples, {train['TARGET'].mean():.2%} default")
print(f"Test:  {len(test):,} samples, {test['TARGET'].mean():.2%} default")

# Get all numeric columns (excluding TARGET and dates)
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
if 'TARGET' in numeric_cols:
    numeric_cols.remove('TARGET')

print(f"\nAnalyzing {len(numeric_cols)} numeric features...")

# Calculate distribution drift for each feature
print("\n[2/5] Calculating Kolmogorov-Smirnov test for distribution drift...")
drift_results = []

for col in numeric_cols:
    # Get non-null values
    train_values = train[col].dropna()
    test_values = test[col].dropna()

    if len(train_values) < 10 or len(test_values) < 10:
        continue

    # KS test: measures distribution difference
    ks_stat, p_value = ks_2samp(train_values, test_values)

    # Calculate missing rate change
    train_missing = train[col].isna().mean()
    test_missing = test[col].isna().mean()
    missing_change = abs(test_missing - train_missing)

    # Calculate mean change
    train_mean = train_values.mean() if len(train_values) > 0 else 0
    test_mean = test_values.mean() if len(test_values) > 0 else 0
    mean_change_pct = abs((test_mean - train_mean) / train_mean * 100) if train_mean != 0 else 0

    # Calculate std change
    train_std = train_values.std() if len(train_values) > 0 else 0
    test_std = test_values.std() if len(test_values) > 0 else 0

    drift_results.append({
        'feature': col,
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'train_missing': train_missing,
        'test_missing': test_missing,
        'missing_change': missing_change,
        'train_mean': train_mean,
        'test_mean': test_mean,
        'mean_change_pct': mean_change_pct,
        'train_std': train_std,
        'test_std': test_std
    })

df_drift = pd.DataFrame(drift_results)

# Identify high drift features
print("\n[3/5] Identifying features with significant drift...")

# Sort by KS statistic (higher = more drift)
df_drift_sorted = df_drift.sort_values('ks_statistic', ascending=False)

print("\n**TOP 20 FEATURES WITH HIGHEST DISTRIBUTION DRIFT:**")
print("(KS Statistic: 0 = identical, 1 = completely different)\n")

for idx, row in df_drift_sorted.head(20).iterrows():
    print(f"{row['feature'][:50]:50s} | KS={row['ks_statistic']:.3f} | "
          f"Mean: {row['train_mean']:8.2f} -> {row['test_mean']:8.2f} ({row['mean_change_pct']:6.1f}%) | "
          f"Missing: {row['train_missing']:.1%} -> {row['test_missing']:.1%}")

# Identify features with major missing rate changes
print("\n[4/5] Identifying features with major missing rate changes...")
missing_drift = df_drift[df_drift['missing_change'] > 0.1].sort_values('missing_change', ascending=False)

if len(missing_drift) > 0:
    print(f"\n**{len(missing_drift)} FEATURES WITH >10% MISSING RATE CHANGE:**")
    print("(These may be new features not available in historical data)\n")

    for idx, row in missing_drift.head(20).iterrows():
        print(f"{row['feature'][:50]:50s} | "
              f"Missing: {row['train_missing']:5.1%} -> {row['test_missing']:5.1%} "
              f"(Change: {row['missing_change']:5.1%})")
else:
    print("\n[OK] No major missing rate changes detected")

# Check for salary and work history features
print("\n[5/5] Checking salary and work history features specifically...")

salary_cols = [c for c in train.columns if 'SALARY' in c.upper() or 'INCOME' in c.upper() or 'WAGE' in c.upper()]
work_cols = [c for c in train.columns if 'WORK' in c.upper() or 'EMPLOY' in c.upper() or 'JOB' in c.upper()]

print(f"\nFound {len(salary_cols)} salary-related columns")
print(f"Found {len(work_cols)} work-related columns")

if salary_cols:
    print("\n**SALARY FEATURES ANALYSIS:**")
    for col in salary_cols[:10]:  # First 10
        if col in df_drift['feature'].values:
            row = df_drift[df_drift['feature'] == col].iloc[0]
            print(f"  {col[:50]:50s} | KS={row['ks_statistic']:.3f} | "
                  f"Missing: {row['train_missing']:.1%} -> {row['test_missing']:.1%}")

if work_cols:
    print("\n**WORK HISTORY FEATURES ANALYSIS:**")
    for col in work_cols[:10]:  # First 10
        if col in df_drift['feature'].values:
            row = df_drift[df_drift['feature'] == col].iloc[0]
            print(f"  {col[:50]:50s} | KS={row['ks_statistic']:.3f} | "
                  f"Missing: {row['train_missing']:.1%} -> {row['test_missing']:.1%}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

high_drift = df_drift[df_drift['ks_statistic'] > 0.3]
print(f"\nFeatures with HIGH drift (KS > 0.3): {len(high_drift)} / {len(df_drift)}")

medium_drift = df_drift[(df_drift['ks_statistic'] > 0.2) & (df_drift['ks_statistic'] <= 0.3)]
print(f"Features with MEDIUM drift (0.2 < KS <= 0.3): {len(medium_drift)} / {len(df_drift)}")

low_drift = df_drift[df_drift['ks_statistic'] <= 0.2]
print(f"Features with LOW drift (KS <= 0.2): {len(low_drift)} / {len(df_drift)}")

print(f"\nFeatures with >10% missing rate change: {len(missing_drift)}")
print(f"Features with >20% mean value change: {len(df_drift[df_drift['mean_change_pct'] > 20])}")

# Save results
df_drift_sorted.to_csv('Output/temporal_drift_analysis.csv', index=False)
print(f"\n[OK] Detailed results saved to: Output/temporal_drift_analysis.csv")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print("\n1. Remove features with KS > 0.4 (severe drift)")
print("2. Investigate features with >20% missing rate change")
print("3. Check if salary/work features are causing the problem")
print("4. Consider time-aware feature engineering")
print("=" * 80)
