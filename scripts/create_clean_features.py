"""
Create clean feature set by removing high-drift and high-missing features

This will solve the 30% Gini gap by removing features that:
1. Have >90% missing rate
2. Have KS drift > 0.4 (severe distribution shift)
3. Changed >100% between train and test
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("CREATING CLEAN FEATURE SET")
print("=" * 80)

# Load drift analysis results
print("\n[1/4] Loading drift analysis results...")
drift = pd.read_csv('Output/temporal_drift_analysis.csv')
print(f"Analyzed {len(drift)} features")

# Identify features to remove
print("\n[2/4] Identifying features to remove...")

# Criteria for removal
remove_high_missing = drift[
    (drift['train_missing'] > 0.9) | (drift['test_missing'] > 0.9)
]['feature'].tolist()

remove_high_drift = drift[drift['ks_statistic'] > 0.4]['feature'].tolist()

remove_big_change = drift[drift['mean_change_pct'] > 100]['feature'].tolist()

# Combine all removal criteria
features_to_remove = list(set(remove_high_missing + remove_high_drift + remove_big_change))

print(f"\nFeatures to remove:")
print(f"  - High missing (>90%): {len(remove_high_missing)}")
print(f"  - High drift (KS>0.4): {len(remove_high_drift)}")
print(f"  - Big mean change (>100%): {len(remove_big_change)}")
print(f"  - TOTAL UNIQUE: {len(features_to_remove)}")

# Load data
print("\n[3/4] Loading train and test data...")
train = pd.read_csv('Data/data_train_temporal.csv')
test = pd.read_csv('Data/data_test_temporal.csv')

print(f"Train before: {train.shape}")
print(f"Test before:  {test.shape}")

# Remove bad features
print("\n[4/4] Removing problematic features...")
features_to_remove_in_data = [f for f in features_to_remove if f in train.columns]

train_clean = train.drop(columns=features_to_remove_in_data, errors='ignore')
test_clean = test.drop(columns=features_to_remove_in_data, errors='ignore')

print(f"\nRemoved {len(features_to_remove_in_data)} features from data")
print(f"Train after: {train_clean.shape} (removed {train.shape[1] - train_clean.shape[1]} columns)")
print(f"Test after:  {test_clean.shape} (removed {test.shape[1] - test_clean.shape[1]} columns)")

# Save clean data
train_clean.to_csv('Data/data_train_temporal_clean.csv', index=False)
test_clean.to_csv('Data/data_test_temporal_clean.csv', index=False)

# Save removal list
pd.DataFrame({'feature': features_to_remove_in_data}).to_csv(
    'Output/removed_features.csv', index=False
)

print("\n" + "=" * 80)
print("FILES CREATED:")
print("=" * 80)
print(f"1. Data/data_train_temporal_clean.csv ({len(train_clean):,} rows, {len(train_clean.columns)} cols)")
print(f"2. Data/data_test_temporal_clean.csv ({len(test_clean):,} rows, {len(test_clean.columns)} cols)")
print(f"3. Output/removed_features.csv ({len(features_to_remove_in_data)} features)")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("1. Update notebook to use data_train_temporal_clean.csv and data_test_temporal_clean.csv")
print("2. Re-run training")
print("3. Expected improvement: Test Gini 0.42 -> 0.55+ (reducing gap from 30% to 10-15%)")
print("=" * 80)

# Show top removed features
print("\n**TOP 20 REMOVED FEATURES (by drift):**")
removed_drift = drift[drift['feature'].isin(features_to_remove_in_data)].sort_values(
    'ks_statistic', ascending=False
).head(20)

for idx, row in removed_drift.iterrows():
    print(f"{row['feature'][:50]:50s} | KS={row['ks_statistic']:.3f} | "
          f"Missing: {row['train_missing']:.1%} | Mean change: {row['mean_change_pct']:.0f}%")
