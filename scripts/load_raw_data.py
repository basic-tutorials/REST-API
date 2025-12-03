"""
Load Raw Data for New Scoring Model
====================================

Simple script to load raw data extracted by raw_data.sql

Performance:
- Loading raw data: ~30 seconds
- Calculating net salary: ~1 second (vs 1.5-2.7 hours in DB)
- Total: <1 minute

Created: 2025-12-03
"""

import pandas as pd
import numpy as np
from typing import Optional
import cx_Oracle
from datetime import datetime
import os


# =====================================================================
# DATABASE CONNECTION
# =====================================================================

def get_db_connection(username: str, password: str, dsn: str) -> cx_Oracle.Connection:
    """Create Oracle database connection"""
    connection = cx_Oracle.connect(username, password, dsn)
    print(f"✓ Connected to database: {dsn}")
    return connection


# =====================================================================
# DATA LOADING
# =====================================================================

def load_raw_data(connection: cx_Oracle.Connection,
                  table_name: str = 'raw_data_final') -> pd.DataFrame:
    """
    Load raw data from database

    Args:
        connection: Oracle database connection
        table_name: Name of the raw data table (default: raw_data_final)

    Returns:
        DataFrame with raw data
    """
    print(f"\n{'='*70}")
    print(f"Loading raw data from {table_name}...")
    print(f"{'='*70}")

    query = f"SELECT * FROM {table_name}"

    start_time = datetime.now()
    df = pd.read_sql(query, connection)
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"✓ Loaded {len(df):,} records in {elapsed:.2f} seconds")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


# =====================================================================
# NET SALARY CALCULATION
# =====================================================================

def calc_net_salary(gross: float, tax_rate: float = 0.14) -> float:
    """
    Calculate net salary from gross salary

    Args:
        gross: Gross salary
        tax_rate: Tax rate (default: 14% for Azerbaijan)

    Returns:
        Net salary
    """
    if pd.isna(gross) or gross == 0:
        return 0.0
    return gross * (1 - tax_rate)


def add_net_salary(df: pd.DataFrame, tax_rate: float = 0.14) -> pd.DataFrame:
    """
    Calculate net salary for all records

    Args:
        df: DataFrame with 'GROSS_SALARY' column
        tax_rate: Tax rate to apply (default: 14%)

    Returns:
        DataFrame with 'NET_SALARY' column
    """
    print(f"\n{'='*70}")
    print("Calculating net salaries...")
    print(f"{'='*70}")

    start_time = datetime.now()

    # Vectorized calculation
    df['NET_SALARY'] = df['GROSS_SALARY'].apply(
        lambda x: calc_net_salary(x, tax_rate)
    )

    elapsed = (datetime.now() - start_time).total_seconds()

    # Statistics
    has_salary = df['GROSS_SALARY'].notna().sum()
    print(f"✓ Calculated in {elapsed:.2f} seconds")
    print(f"  Records with salary: {has_salary:,} ({has_salary/len(df)*100:.1f}%)")
    print(f"  Tax rate applied: {tax_rate*100:.0f}%")
    print(f"  Avg gross salary: {df['GROSS_SALARY'].mean():.2f}")
    print(f"  Avg net salary: {df['NET_SALARY'].mean():.2f}")

    return df


# =====================================================================
# DATA QUALITY CHECKS
# =====================================================================

def show_data_summary(df: pd.DataFrame) -> None:
    """Print data quality summary"""

    print(f"\n{'='*70}")
    print("DATA SUMMARY")
    print(f"{'='*70}")

    # Basic stats
    print(f"\nRecords: {len(df):,}")
    print(f"Unique customers (ID): {df['ID'].nunique():,}")
    print(f"Unique FINs: {df['FIN'].nunique():,}")

    # Date range
    print(f"\nDate range:")
    print(f"  From: {df['MKR_DATE'].min()}")
    print(f"  To:   {df['MKR_DATE'].max()}")

    # Salary coverage
    pct_salary = (df['GROSS_SALARY'].notna().sum() / len(df)) * 100
    print(f"\nSalary coverage: {pct_salary:.1f}%")

    # Top credit types
    if 'CREDIT_TYPE_NAME' in df.columns:
        print(f"\nTop 5 credit types:")
        for ctype, count in df['CREDIT_TYPE_NAME'].value_counts().head().items():
            pct = count / len(df) * 100
            print(f"  {ctype}: {count:,} ({pct:.1f}%)")

    # Top banks
    if 'BANK_NAME' in df.columns:
        print(f"\nTop 5 banks:")
        for bank, count in df['BANK_NAME'].value_counts().head().items():
            pct = count / len(df) * 100
            print(f"  {bank}: {count:,} ({pct:.1f}%)")

    # Credit status
    if 'CREDIT_STATUS' in df.columns:
        print(f"\nCredit status distribution:")
        for status, count in df['CREDIT_STATUS'].value_counts().head().items():
            pct = count / len(df) * 100
            print(f"  {status}: {count:,} ({pct:.1f}%)")


# =====================================================================
# EXPORT
# =====================================================================

def export_to_csv(df: pd.DataFrame, output_path: str = 'Data/raw_data.csv') -> None:
    """Export data to CSV"""

    print(f"\n{'='*70}")
    print(f"Exporting to {output_path}...")
    print(f"{'='*70}")

    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    start_time = datetime.now()
    df.to_csv(output_path, index=False)
    elapsed = (datetime.now() - start_time).total_seconds()

    file_size_mb = os.path.getsize(output_path) / 1024**2

    print(f"✓ Exported {len(df):,} records in {elapsed:.2f} seconds")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")


# =====================================================================
# MAIN
# =====================================================================

def main():
    """Main execution"""

    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║           Raw Data Loading for New Scoring Model                ║
    ║                                                                  ║
    ║  Fast extraction: ~3-5 minutes (vs 4-5 hours old approach)      ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Database connection parameters
    # UPDATE THESE WITH YOUR CREDENTIALS
    USERNAME = 'your_username'
    PASSWORD = 'your_password'
    DSN = 'your_host:1521/your_service'

    try:
        # Connect to database
        print("Connecting to database...")
        connection = get_db_connection(USERNAME, PASSWORD, DSN)

        # Load raw data
        df = load_raw_data(connection, table_name='raw_data_final')

        # Calculate net salary
        df = add_net_salary(df, tax_rate=0.14)  # Adjust tax rate as needed

        # Show summary
        show_data_summary(df)

        # Export to CSV
        export_to_csv(df, 'Data/raw_data.csv')

        # Close connection
        connection.close()
        print("\n✓ Database connection closed")

    except cx_Oracle.Error as e:
        print(f"\n✗ Database error: {e}")
        print("\nMake sure:")
        print("  1. Oracle client is installed (cx_Oracle)")
        print("  2. Connection parameters are correct")
        print("  3. You have access to the tables")
        print("  4. You ran raw_data.sql first to create tables")
        return

    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print("\nData saved to: Data/raw_data.csv")
    print("\nNext steps:")
    print("  1. Use raw_data.csv for your feature engineering")
    print("  2. Build features in Python (faster & more flexible)")
    print("  3. Train your new scoring model")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
