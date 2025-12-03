"""
Fix train/test split for credit scoring model

PROBLEMS WITH CURRENT SPLIT:
1. Train and test have same date range (2018-2021)
2. Test has 2X higher default rate (3.23% vs 1.66%)
3. Not a proper temporal split for credit model

SOLUTION:
Create proper temporal split:
- Train: Earlier loans (used to build model)
- Test: Later loans (simulate real-world deployment)
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("FIXING TRAIN/TEST SPLIT FOR CREDIT SCORING")
print("=" * 70)

# Load data
print("\n[1/4] Loading data...")
df = pd.read_csv('Data/data.csv')
print(f"[OK] Loaded {df.shape[0]:,} rows, {df.shape[1]:,} columns")

# Check date distribution
print("\n[2/4] Analyzing date distribution...")
df['MKR_DATE'] = pd.to_datetime(df['MKR_DATE'])
df['YEAR'] = df['MKR_DATE'].dt.year
df['YEAR_MONTH'] = df['MKR_DATE'].dt.to_period('M')

print("\nDefault rate by year:")
for year in sorted(df['YEAR'].unique()):
    year_data = df[df['YEAR'] == year]
    default_rate = year_data['TARGET'].mean()
    count = len(year_data)
    defaults = year_data['TARGET'].sum()
    print(f"  {year}: {count:,} loans, {int(defaults):,} defaults ({default_rate:.2%})")

print("\nDefault rate by month:")
monthly_stats = df.groupby('YEAR_MONTH').agg({
    'TARGET': ['count', 'sum', 'mean']
}).round(4)
print(monthly_stats.tail(12))  # Last 12 months

# Recommend split point
print("\n[3/4] Recommending split strategy...")
print("\n**OPTION 1: Time-based split (RECOMMENDED for production)**")
print("  Train: 2018-2019 data")
print("  Test:  2020-2021 data")
print("  Models real-world scenario: predict future from past")

train_cutoff_1 = pd.Timestamp('2020-01-01')
train_1 = df[df['MKR_DATE'] < train_cutoff_1]
test_1 = df[df['MKR_DATE'] >= train_cutoff_1]
print(f"  Train: {len(train_1):,} loans, default rate: {train_1['TARGET'].mean():.2%}")
print(f"  Test:  {len(test_1):,} loans, default rate: {test_1['TARGET'].mean():.2%}")

print("\n**OPTION 2: Time-based split (More balanced)**")
print("  Train: 2018-mid 2020 data")
print("  Test:  mid 2020-2021 data")

train_cutoff_2 = pd.Timestamp('2020-07-01')
train_2 = df[df['MKR_DATE'] < train_cutoff_2]
test_2 = df[df['MKR_DATE'] >= train_cutoff_2]
print(f"  Train: {len(train_2):,} loans, default rate: {train_2['TARGET'].mean():.2%}")
print(f"  Test:  {len(test_2):,} loans, default rate: {test_2['TARGET'].mean():.2%}")

print("\n**OPTION 3: Stratified random split (ONLY if temporal not possible)**")
print("  Ensures same default rate in train/test")
print("  But mixes time periods (less realistic for deployment)")

from sklearn.model_selection import train_test_split
train_3, test_3 = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['TARGET']
)
print(f"  Train: {len(train_3):,} loans, default rate: {train_3['TARGET'].mean():.2%}")
print(f"  Test:  {len(test_3):,} loans, default rate: {test_3['TARGET'].mean():.2%}")
print(f"  Train dates: {train_3['MKR_DATE'].min()} to {train_3['MKR_DATE'].max()}")
print(f"  Test dates:  {test_3['MKR_DATE'].min()} to {test_3['MKR_DATE'].max()}")

# Save recommended split
print("\n[4/4] Saving recommended split (Option 2)...")
train_2.to_csv('Data/data_train_temporal.csv', index=False)
test_2.to_csv('Data/data_test_temporal.csv', index=False)
print(f"[OK] Saved Data/data_train_temporal.csv ({len(train_2):,} rows)")
print(f"[OK] Saved Data/data_test_temporal.csv ({len(test_2):,} rows)")

print("\n" + "=" * 70)
print("RECOMMENDED NEXT STEPS:")
print("=" * 70)
print("1. Choose Option 2 (time-based split) for production model")
print("2. Update notebook to use data_train_temporal.csv and data_test_temporal.csv")
print("3. Re-train models with proper temporal split")
print("4. Also fix Layer 2 data leakage issue")
print("=" * 70)
