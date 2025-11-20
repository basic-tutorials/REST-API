#!/usr/bin/env python3
"""
Sample Data Generator for Bank of Baku Credit Scoring Pipeline

This script generates synthetic loan application data that matches the expected
format for the credit scoring pipeline. Use this for testing and development.

Usage:
    python generate_sample_data.py

Output:
    Data/sample_train_data.csv - Training dataset (10,000 samples)
    Data/sample_test_data.csv  - Test dataset (2,500 samples)
    Data/sample_score_data.csv - Scoring dataset (1,000 samples)
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)


def generate_credit_bureau_features(n_samples, default_rate=0.05):
    """
    Generate synthetic credit bureau features.

    Features follow the naming convention: {ProductType}_{Status}_{TimePeriod}_{Metric}

    Args:
        n_samples: Number of samples to generate
        default_rate: Expected default rate (affects feature distributions)

    Returns:
        dict: Dictionary of feature arrays
    """
    features = {}

    # Generate target first (affects feature distributions)
    target = np.random.binomial(1, default_rate, n_samples)

    # Credit Card Features (CC)
    # Worst Payment Status: 0=Good, 1-5=Increasing severity
    features['CC_O_3MWPS_EVER'] = np.where(
        target == 1,
        np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.3, 0.2, 0.2]),
        np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.08, 0.02])
    )

    features['CC_C_3MWPS_90D'] = np.where(
        target == 1,
        np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.35, 0.25]),
        np.random.choice([0, 1, 2], n_samples, p=[0.8, 0.15, 0.05])
    )

    features['CC_A_CWPS_90D'] = np.random.randint(0, 4, n_samples)
    features['CC_O_EVERWPS_90D'] = np.random.randint(0, 5, n_samples)

    # Consumer Loan Features (CL)
    features['CL_A_CWPS_90D'] = np.where(
        target == 1,
        np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
        np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.08, 0.02])
    )

    features['CL_O_3MWPS_90D'] = np.random.randint(0, 4, n_samples)
    features['CL_C_EVERWPS_90D'] = np.random.randint(0, 3, n_samples)
    features['CL_A_4M6MWPS_90D'] = np.random.randint(0, 4, n_samples)

    # Housing Loan Features (HL)
    features['HL_O_3MWPS_90D'] = np.where(
        target == 1,
        np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]),
        np.random.choice([0, 1, 2], n_samples, p=[0.85, 0.12, 0.03])
    )

    features['HL_C_CWPS_90D'] = np.random.randint(0, 3, n_samples)
    features['HL_A_7M12MWPS_90D'] = np.random.randint(0, 4, n_samples)
    features['HL_O_13M24MWPS_90D'] = np.random.randint(0, 3, n_samples)

    # Other Loan Features (OL)
    features['OL_O_3MWPS_90D'] = np.random.randint(0, 4, n_samples)
    features['OL_C_CWPS_90D'] = np.random.randint(0, 3, n_samples)
    features['OL_A_EVERWPS_90D'] = np.where(
        target == 1,
        np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.3, 0.2, 0.2]),
        np.random.choice([0, 1, 2, 3], n_samples, p=[0.75, 0.18, 0.05, 0.02])
    )

    # All Products Combined (ALL)
    features['ALL_O_3MWPS_90D'] = np.random.randint(0, 5, n_samples)
    features['ALL_C_CWPS_90D'] = np.random.randint(0, 4, n_samples)
    features['ALL_A_4M6MWPS_90D'] = np.random.randint(0, 4, n_samples)
    features['ALL_O_7M12MWPS_90D'] = np.random.randint(0, 3, n_samples)

    # Credit Card + Other Loan (CCOL)
    features['CCOL_O_3MWPS_90D'] = np.random.randint(0, 4, n_samples)
    features['CCOL_C_EVERWPS_90D'] = np.random.randint(0, 3, n_samples)

    # Utilization and Limit Features (continuous)
    features['CC_O_UTLR_90D'] = np.where(
        target == 1,
        np.random.beta(3, 2, n_samples),  # Higher utilization for defaults
        np.random.beta(2, 3, n_samples)   # Lower utilization for non-defaults
    )

    features['CL_O_LMTSUM_90D'] = np.random.lognormal(10, 1.5, n_samples)
    features['HL_O_UTLR_90D'] = np.random.beta(2, 5, n_samples)
    features['ALL_O_LMTAVG_90D'] = np.random.lognormal(9, 1.2, n_samples)

    return features, target


def generate_demographic_features(n_samples, target):
    """
    Generate synthetic demographic features.

    Args:
        n_samples: Number of samples to generate
        target: Target array (affects feature distributions)

    Returns:
        dict: Dictionary of feature arrays
    """
    features = {}

    # Age (25-70 years)
    features['AGE'] = np.where(
        target == 1,
        np.random.normal(35, 10, n_samples).clip(25, 70),  # Younger defaults
        np.random.normal(42, 12, n_samples).clip(25, 70)   # Older non-defaults
    ).astype(int)

    # Gender (0=Female, 1=Male)
    features['GENDER'] = np.random.binomial(1, 0.52, n_samples)

    # Education Level (1=High School, 2=Bachelor, 3=Master, 4=PhD)
    features['EDUCATION'] = np.where(
        target == 1,
        np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.4, 0.15, 0.05]),
        np.random.choice([1, 2, 3, 4], n_samples, p=[0.25, 0.45, 0.22, 0.08])
    )

    # Employment Status (0=Unemployed, 1=Employed, 2=Self-employed, 3=Retired)
    features['EMPLOYMENT_STATUS'] = np.where(
        target == 1,
        np.random.choice([0, 1, 2, 3], n_samples, p=[0.15, 0.55, 0.20, 0.10]),
        np.random.choice([0, 1, 2, 3], n_samples, p=[0.05, 0.65, 0.22, 0.08])
    )

    # Monthly Income (AZN)
    features['INCOME'] = np.where(
        target == 1,
        np.random.lognormal(7.5, 0.6, n_samples),  # Lower income for defaults
        np.random.lognormal(8.2, 0.7, n_samples)   # Higher income for non-defaults
    ).clip(500, 50000).astype(int)

    # Marital Status (0=Single, 1=Married, 2=Divorced, 3=Widowed)
    features['MARITAL_STATUS'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.30, 0.55, 0.10, 0.05])

    # Number of Dependents
    features['DEPENDENTS'] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.25, 0.30, 0.25, 0.15, 0.05])

    # Region (1-10 representing different geographical areas)
    features['REGION'] = np.random.randint(1, 11, n_samples)

    return features


def generate_application_features(n_samples, target, income):
    """
    Generate synthetic loan application features.

    Args:
        n_samples: Number of samples to generate
        target: Target array (affects feature distributions)
        income: Income array (affects loan amounts)

    Returns:
        dict: Dictionary of feature arrays
    """
    features = {}

    # Requested Loan Amount (correlated with income)
    features['LOAN_AMOUNT'] = (income * np.random.uniform(0.5, 3.0, n_samples)).clip(1000, 100000).astype(int)

    # Loan Purpose (1=Consumer, 2=Housing, 3=Auto, 4=Business, 5=Other)
    features['LOAN_PURPOSE'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.35, 0.25, 0.20, 0.12, 0.08])

    # Collateral Type (0=None, 1=Real Estate, 2=Vehicle, 3=Other)
    features['COLLATERAL_TYPE'] = np.where(
        target == 1,
        np.random.choice([0, 1, 2, 3], n_samples, p=[0.50, 0.25, 0.15, 0.10]),
        np.random.choice([0, 1, 2, 3], n_samples, p=[0.30, 0.40, 0.20, 0.10])
    )

    # Loan Term (months)
    features['LOAN_TERM'] = np.random.choice([12, 24, 36, 48, 60, 84, 120], n_samples,
                                             p=[0.15, 0.25, 0.25, 0.15, 0.10, 0.05, 0.05])

    # Debt-to-Income Ratio
    features['DEBT_TO_INCOME'] = np.where(
        target == 1,
        np.random.beta(4, 2, n_samples),  # Higher DTI for defaults
        np.random.beta(2, 4, n_samples)   # Lower DTI for non-defaults
    ).clip(0, 1)

    # Previous Loans Count
    features['PREVIOUS_LOANS'] = np.random.poisson(2, n_samples).clip(0, 10)

    # Account Age (months with bank)
    features['ACCOUNT_AGE_MONTHS'] = np.where(
        target == 1,
        np.random.exponential(24, n_samples).clip(1, 240),
        np.random.exponential(36, n_samples).clip(1, 240)
    ).astype(int)

    return features


def add_missing_values(df, missing_rate=0.02):
    """
    Add random missing values to simulate real-world data.

    Args:
        df: DataFrame to modify
        missing_rate: Proportion of values to set as missing

    Returns:
        pd.DataFrame: DataFrame with missing values
    """
    # Don't add missing values to MUQAVILE or TARGET
    columns_to_modify = [col for col in df.columns if col not in ['MUQAVILE', 'TARGET']]

    for col in columns_to_modify:
        # Some features have higher missing rates
        col_missing_rate = missing_rate
        if 'EVERWPS' in col or 'HL_' in col:
            col_missing_rate = missing_rate * 3  # 6% missing for some features

        n_missing = int(len(df) * col_missing_rate)
        if n_missing > 0:
            missing_idx = np.random.choice(df.index, n_missing, replace=False)
            df.loc[missing_idx, col] = np.nan

    return df


def generate_dataset(n_samples, default_rate=0.05, start_id=1):
    """
    Generate complete synthetic credit scoring dataset.

    Args:
        n_samples: Number of samples to generate
        default_rate: Expected default rate
        start_id: Starting contract ID number

    Returns:
        pd.DataFrame: Complete dataset with all features
    """
    print(f"Generating {n_samples} samples with {default_rate:.1%} default rate...")

    # Generate features
    credit_features, target = generate_credit_bureau_features(n_samples, default_rate)
    demographic_features = generate_demographic_features(n_samples, target)
    application_features = generate_application_features(n_samples, target, demographic_features['INCOME'])

    # Combine all features
    all_features = {}
    all_features.update(credit_features)
    all_features.update(demographic_features)
    all_features.update(application_features)

    # Create DataFrame
    df = pd.DataFrame(all_features)

    # Add contract IDs
    df.insert(0, 'MUQAVILE', [f'CONT_{start_id + i:06d}' for i in range(n_samples)])

    # Add target
    df['TARGET'] = target

    # Add missing values
    df = add_missing_values(df, missing_rate=0.02)

    print(f"  - Shape: {df.shape}")
    print(f"  - Default rate: {df['TARGET'].mean():.2%}")
    print(f"  - Missing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / (df.shape[0] * df.shape[1]):.2%})")

    return df


def main():
    """Generate all sample datasets."""
    print("="*60)
    print("Bank of Baku Credit Scoring - Sample Data Generator")
    print("="*60)
    print()

    # Create Data directory if it doesn't exist
    os.makedirs('Data', exist_ok=True)

    # Generate training data (larger, lower default rate)
    print("1. Generating Training Data")
    train_df = generate_dataset(
        n_samples=10000,
        default_rate=0.045,
        start_id=1
    )
    train_path = 'Data/sample_train_data.csv'
    train_df.to_csv(train_path, index=False)
    print(f"   Saved to: {train_path}")
    print()

    # Generate test/validation data
    print("2. Generating Test/Validation Data")
    test_df = generate_dataset(
        n_samples=2500,
        default_rate=0.048,
        start_id=10001
    )
    test_path = 'Data/sample_test_data.csv'
    test_df.to_csv(test_path, index=False)
    print(f"   Saved to: {test_path}")
    print()

    # Generate new scoring data (no target, will be filled with 0s for structure)
    print("3. Generating New Applicant Scoring Data")
    score_df = generate_dataset(
        n_samples=1000,
        default_rate=0.05,
        start_id=20001
    )
    # In real scoring, we wouldn't have target, but include it for validation
    score_path = 'Data/sample_score_data.csv'
    score_df.to_csv(score_path, index=False)
    print(f"   Saved to: {score_path}")
    print()

    # Generate summary statistics
    print("="*60)
    print("Dataset Summary")
    print("="*60)
    print(f"Training:   {len(train_df):,} samples, {train_df['TARGET'].sum():,} defaults ({train_df['TARGET'].mean():.2%})")
    print(f"Test:       {len(test_df):,} samples, {test_df['TARGET'].sum():,} defaults ({test_df['TARGET'].mean():.2%})")
    print(f"Scoring:    {len(score_df):,} samples, {score_df['TARGET'].sum():,} defaults ({score_df['TARGET'].mean():.2%})")
    print()
    print(f"Total features: {train_df.shape[1] - 1} (excluding TARGET)")
    print(f"Feature types:")
    print(f"  - Credit Bureau: {len([c for c in train_df.columns if any(x in c for x in ['CC_', 'CL_', 'HL_', 'OL_', 'ALL_', 'CCOL_'])])} features")
    print(f"  - Demographic: {len([c for c in train_df.columns if c in ['AGE', 'GENDER', 'EDUCATION', 'EMPLOYMENT_STATUS', 'INCOME', 'MARITAL_STATUS', 'DEPENDENTS', 'REGION']])} features")
    print(f"  - Application: {len([c for c in train_df.columns if c in ['LOAN_AMOUNT', 'LOAN_PURPOSE', 'COLLATERAL_TYPE', 'LOAN_TERM', 'DEBT_TO_INCOME', 'PREVIOUS_LOANS', 'ACCOUNT_AGE_MONTHS']])} features")
    print()

    # Show sample records
    print("Sample Records (first 3 from training):")
    print(train_df[['MUQAVILE', 'AGE', 'INCOME', 'LOAN_AMOUNT', 'TARGET']].head(3).to_string(index=False))
    print()

    print("="*60)
    print("✓ Sample data generation complete!")
    print("="*60)
    print()
    print("Next steps:")
    print("  1. Review generated CSV files in Data/ directory")
    print("  2. Use these files for pipeline testing and development")
    print("  3. Replace with real data for production training")


if __name__ == "__main__":
    main()
