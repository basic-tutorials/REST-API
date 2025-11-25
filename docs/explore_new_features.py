#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Exploration Script for New Features
Analyzes MAAS, WORKGROUP, QRUP, SAHƏLƏR, SEKTOR, PARTNYORLUQ
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXPLORING NEW FEATURES IN data.csv")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv('Data/data.csv')
print(f"Data loaded: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

# Set index
df = df.set_index('MUQAVILE')

# Check if TARGET exists
has_target = 'TARGET' in df.columns
print(f"TARGET column present: {has_target}")

print("\n" + "="*80)
print("1. MAAS (Salary/Income) - NUMERIC FEATURE")
print("="*80)
print("\nDescriptive Statistics:")
print(df['MAAS'].describe())
print(f"\nMissing values: {df['MAAS'].isna().sum():,} ({df['MAAS'].isna().mean():.1%})")
print(f"Unique values: {df['MAAS'].nunique():,}")

if has_target:
    # Calculate default rate by salary quantiles
    df['MAAS_QUARTILE'] = pd.qcut(df['MAAS'], q=4, labels=['Q1-Low', 'Q2', 'Q3', 'Q4-High'], duplicates='drop')
    default_by_salary = df.groupby('MAAS_QUARTILE')['TARGET'].agg(['mean', 'count'])
    print("\nDefault Rate by Salary Quartile:")
    print(default_by_salary)

    # Correlation
    corr = df[['MAAS', 'TARGET']].corr().iloc[0, 1]
    print(f"\nCorrelation with TARGET: {corr:.4f}")

print("\n" + "="*80)
print("2. WORKGROUP (Employment Type) - CATEGORICAL FEATURE")
print("="*80)
print("\nValue Counts:")
print(df['WORKGROUP'].value_counts(dropna=False))
print(f"\nMissing values: {df['WORKGROUP'].isna().sum():,} ({df['WORKGROUP'].isna().mean():.1%})")

if has_target:
    print("\nDefault Rate by WORKGROUP:")
    default_by_workgroup = df.groupby('WORKGROUP', dropna=False)['TARGET'].agg(['mean', 'count'])
    print(default_by_workgroup)

print("\n" + "="*80)
print("3. QRUP (Organization Type) - CATEGORICAL FEATURE")
print("="*80)
print("\nValue Counts:")
print(df['QRUP'].value_counts(dropna=False))
print(f"\nMissing values: {df['QRUP'].isna().sum():,} ({df['QRUP'].isna().mean():.1%})")

if has_target:
    print("\nDefault Rate by QRUP:")
    default_by_qrup = df.groupby('QRUP', dropna=False)['TARGET'].agg(['mean', 'count'])
    print(default_by_qrup)

print("\n" + "="*80)
print("4. SAHƏLƏR (Business Sector) - CATEGORICAL FEATURE")
print("="*80)
print("\nValue Counts (Top 15):")
print(df['SAHƏLƏR'].value_counts(dropna=False).head(15))
print(f"\nTotal unique values: {df['SAHƏLƏR'].nunique()}")
print(f"Missing values: {df['SAHƏLƏR'].isna().sum():,} ({df['SAHƏLƏR'].isna().mean():.1%})")

if has_target:
    print("\nDefault Rate by SAHƏLƏR (Top 10 sectors):")
    default_by_sector = df.groupby('SAHƏLƏR', dropna=False)['TARGET'].agg(['mean', 'count']).sort_values('count', ascending=False).head(10)
    print(default_by_sector)

print("\n" + "="*80)
print("5. SEKTOR (Specific Sector) - CATEGORICAL FEATURE")
print("="*80)
print("\nValue Counts (Top 15):")
print(df['SEKTOR'].value_counts(dropna=False).head(15))
print(f"\nTotal unique values: {df['SEKTOR'].nunique()}")
print(f"Missing values: {df['SEKTOR'].isna().sum():,} ({df['SEKTOR'].isna().mean():.1%})")

if has_target:
    print("\nDefault Rate by SEKTOR (Top 10):")
    default_by_sektor = df.groupby('SEKTOR', dropna=False)['TARGET'].agg(['mean', 'count']).sort_values('count', ascending=False).head(10)
    print(default_by_sektor)

print("\n" + "="*80)
print("6. PARTNYORLUQ (Partnership Type) - CATEGORICAL FEATURE")
print("="*80)
print("\nValue Counts:")
print(df['PARTNYORLUQ'].value_counts(dropna=False))
print(f"\nMissing values: {df['PARTNYORLUQ'].isna().sum():,} ({df['PARTNYORLUQ'].isna().mean():.1%})")

if has_target:
    print("\nDefault Rate by PARTNYORLUQ:")
    default_by_part = df.groupby('PARTNYORLUQ', dropna=False)['TARGET'].agg(['mean', 'count'])
    print(default_by_part)

print("\n" + "="*80)
print("7. CORRELATION BETWEEN SAHƏLƏR AND SEKTOR")
print("="*80)
# Check if they're redundant
if df['SAHƏLƏR'].nunique() > 0 and df['SEKTOR'].nunique() > 0:
    # Count unique combinations
    unique_combinations = df.groupby(['SAHƏLƏR', 'SEKTOR']).size().reset_index(name='count')
    print(f"\nUnique SAHƏLƏR values: {df['SAHƏLƏR'].nunique()}")
    print(f"Unique SEKTOR values: {df['SEKTOR'].nunique()}")
    print(f"Unique combinations: {len(unique_combinations)}")

    print("\nTop 10 SAHƏLƏR-SEKTOR combinations:")
    print(unique_combinations.sort_values('count', ascending=False).head(10))

print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

recommendations = []

# MAAS
if df['MAAS'].isna().mean() < 0.3:
    recommendations.append("✅ MAAS: Low missing rate - INCLUDE (strong predictor expected)")
else:
    recommendations.append(f"⚠️ MAAS: High missing rate ({df['MAAS'].isna().mean():.1%}) - INCLUDE but impute carefully")

# WORKGROUP
if df['WORKGROUP'].nunique() > 1 and df['WORKGROUP'].isna().mean() < 0.5:
    recommendations.append("✅ WORKGROUP: Good variation - INCLUDE")
else:
    recommendations.append("⚠️ WORKGROUP: Low variation or high missing rate - CONSIDER DROPPING")

# QRUP
if df['QRUP'].nunique() > 1:
    recommendations.append("✅ QRUP: INCLUDE")
else:
    recommendations.append("❌ QRUP: Only one value - DROP (constant)")

# SAHƏLƏR
if df['SAHƏLƏR'].nunique() > 5:
    recommendations.append(f"✅ SAHƏLƏR: {df['SAHƏLƏR'].nunique()} unique sectors - INCLUDE (use target encoding)")
else:
    recommendations.append("⚠️ SAHƏLƏR: Low cardinality - INCLUDE with caution")

# SEKTOR vs SAHƏLƏR
saheler_unique = df['SAHƏLƏR'].nunique()
sektor_unique = df['SEKTOR'].nunique()
if abs(saheler_unique - sektor_unique) < 5:
    recommendations.append("⚠️ SEKTOR vs SAHƏLƏR: Highly redundant - CHOOSE ONE (prefer SAHƏLƏR)")
else:
    recommendations.append("✅ SEKTOR: Different from SAHƏLƏR - INCLUDE BOTH")

# PARTNYORLUQ
if df['PARTNYORLUQ'].nunique() <= 2:
    recommendations.append("⚠️ PARTNYORLUQ: Very low variation - CONSIDER DROPPING")
else:
    recommendations.append("✅ PARTNYORLUQ: INCLUDE")

print("\n")
for rec in recommendations:
    print(rec)

print("\n" + "="*80)
print("FINAL RECOMMENDED FEATURES TO USE:")
print("="*80)
print("""
DEFINITE YES:
- MAAS (salary) - Strong income predictor
- WORKGROUP (employment type) - Stability indicator
- QRUP (organization type) - Job sector
- SAHƏLƏR (business sector) - Industry risk

MAYBE:
- SEKTOR (specific sector) - If not too correlated with SAHƏLƏR
- PARTNYORLUQ (partnership) - If has >2 unique values

DEFINITE NO:
- ID, FIN, MKR_ID, MKR_DATE, IDS - These are ID columns
- SCR_CONSTANT, SCR_MULTIPLIER - Policy parameters, not predictive
""")

print("\n" + "="*80)
print("Exploration complete! Use these insights to update cols_to_drop in the training notebook.")
print("="*80)
