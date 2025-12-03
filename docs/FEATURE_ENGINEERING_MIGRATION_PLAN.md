# Feature Engineering Migration Plan
## From SQL to Python Pipeline Integration

**Document Version:** 1.0
**Date:** 2025-11-26
**Status:** Planning Phase

---

## Executive Summary

This document outlines the migration of 1000+ feature engineering transformations from SQL (scripts 1-6, ~5,148 lines) into the existing Python pipeline (`QNBAnalytics_ML` package). The goal is to achieve a production-ready, reproducible, and maintainable ML pipeline.

**Current Problem:**
- Feature engineering happens in SQL (scripts/3.sql - 5,148 lines)
- Scoring new applications requires re-running entire SQL pipeline
- Train/test consistency not guaranteed
- Hard to experiment with new features
- Feature logic separated from model code

**Target Solution:**
- Extract only raw transactional data from SQL
- Perform all feature engineering in Python
- Integrate with existing `QNBAnalytics_ML` pipeline classes
- Enable real-time scoring of new applications

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Target Architecture](#2-target-architecture)
3. [Migration Strategy](#3-migration-strategy)
4. [Implementation Phases](#4-implementation-phases)
5. [Integration with Existing Code](#5-integration-with-existing-code)
6. [Testing & Validation](#6-testing--validation)
7. [Deployment Plan](#7-deployment-plan)
8. [Risk Mitigation](#8-risk-mitigation)

---

## 1. Current State Analysis

### 1.1 SQL Pipeline Overview

**Script 1 (1.sql):** Raw data extraction
- Source: `scoring.qnb_mkr_data_mba_backup`
- Output: `MKR_DATA` table (loan-level transactions)
- ~30 columns: loan details, payment history, amounts, dates

**Script 2 (2.sql):** Salary enrichment
- Source: `ibs.asan_finance_is_yeri_v2@ibs_ro`
- Output: `YEKUN_FOR_DATAMART` table
- Adds: Net salary calculation

**Script 3 (3.sql):** Feature engineering ⚠️ **PROBLEM AREA**
- **5,148 lines** of SQL aggregations
- Creates 16 temporary tables
- Generates 1000+ features:
  - Count features (e.g., `CL_OCNT_90D`)
  - Amount features (e.g., `CC_OSMTOB_EVER`)
  - Utilization features (e.g., `CC_OLMTUTL_182D`)
  - Payment status encodings
  - Score adjustment codes
- Output: `MKR_LONGLIST_*` tables

**Script 4-5:** MKR ID matching
- Links credit bureau reports to applications

**Script 6 (6.sql):** Final assembly
- Joins all features into `fourth_table`
- Output: 1000+ column dataset
- Exported to: `Data/data.csv`

### 1.2 Python Pipeline Overview

**Existing Classes in `QNBAnalytics_ML/`:**

| Module | Class | Purpose | Integration Point |
|--------|-------|---------|-------------------|
| `preprocessing.py` | `conversions()` | Type conversions | ✅ Keep as-is |
| `preprocessing.py` | `impute_null()` | Missing value imputation | ✅ Keep as-is |
| `preprocessing.py` | `encode()` | Target encoding | ✅ Keep as-is |
| `preprocessing.py` | `scale()` | Standardization | ✅ Keep as-is |
| `elimination.py` | `DropNullFeatures` | Remove high-null features | ✅ Keep as-is |
| `elimination.py` | `DropConstantFeatures` | Remove zero-variance | ✅ Keep as-is |
| `elimination.py` | `DropLowGiniFeatures` | Remove low predictive power | ✅ Keep as-is |
| `elimination.py` | `DropCorrelatedLowerGiniFeatures` | Remove redundant | ✅ Keep as-is |
| `binning.py` | `WOEBinning` | WOE encoding | ✅ Keep as-is |
| `classifier.py` | `LogisticRegression`, `LightGBM`, etc. | Models | ✅ Keep as-is |

**Current Flow:**
```
data.csv (pre-engineered) → elimination → WOE binning → scaling → model
```

**Target Flow:**
```
data_raw.csv → FEATURE ENGINEERING → elimination → WOE binning → scaling → model
                      ↑ NEW MODULE
```

---

## 2. Target Architecture

### 2.1 New Module Structure

Create new module: `QNBAnalytics_ML/feature_engineering.py`

```python
QNBAnalytics_ML/
├── feature_engineering.py         # ← NEW MODULE
│   ├── CurrencyConverter
│   ├── PaymentStatusEncoder
│   ├── ProductTypeClassifier
│   ├── CreditBureauAggregator
│   ├── UtilizationCalculator
│   ├── ScoreCodeCalculator
│   └── FeatureEngineeringPipeline  # Main orchestrator
├── preprocessing.py                # Existing - no changes
├── elimination.py                  # Existing - no changes
├── binning.py                      # Existing - no changes
├── classifier.py                   # Existing - no changes
└── helpers.py                      # Existing - may add utilities
```

### 2.2 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA EXTRACTION (SQL - Simplified)                    │
├─────────────────────────────────────────────────────────────────┤
│ • Extract raw loan transactions (~30 columns)                  │
│ • Person-level: FIN, MKR_DATE, SALARY                         │
│ • Loan-level: credit_type, amounts, dates, payment_history    │
│ • Output: data_raw.csv (~30 columns × 500K rows loan-level)   │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: FEATURE ENGINEERING (Python - NEW)                    │
├─────────────────────────────────────────────────────────────────┤
│ Step 1: CurrencyConverter                                       │
│   • Convert USD/EUR/RUB to AZN                                 │
│   • USD×1.7, EUR×1.9, RUB×0.025                                │
│                                                                 │
│ Step 2: ProductTypeClassifier                                   │
│   • Classify loans: CL, CC, HL, OL, CHL, CCOL, ALL            │
│                                                                 │
│ Step 3: PaymentStatusEncoder                                    │
│   • Generate payment status strings (e.g., "0001239XX")        │
│                                                                 │
│ Step 4: CreditBureauAggregator                                  │
│   • Group by (FIN, MKR_DATE)                                   │
│   • Aggregate features across:                                 │
│     - Product types (CL, CC, HL, OL, CHL, CCOL, ALL)         │
│     - Time periods (EVER, 30D, 90D, 182D, 365D)               │
│     - Account status (Open, Closed, All)                       │
│     - Payment status (CWPS0, CWPS1_6, CWPS6P)                 │
│   • Create count, sum, avg, max features                       │
│                                                                 │
│ Step 5: UtilizationCalculator                                   │
│   • Calculate credit utilization ratios                        │
│   • Debt-to-limit, payment-to-balance ratios                  │
│                                                                 │
│ Step 6: ScoreCodeCalculator                                     │
│   • Calculate SCR_CONSTANT, SCR_MULTIPLIER                     │
│   • Based on delinquency severity rules                        │
│                                                                 │
│ Output: data_features.csv (~1000 columns × 96K rows)          │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: FEATURE SELECTION (Python - Existing)                 │
├─────────────────────────────────────────────────────────────────┤
│ • DropNullFeatures (>99% missing)                              │
│ • DropConstantFeatures (zero variance)                         │
│ • DropLowGiniFeatures (Gini < 0.05)                           │
│ • DropCorrelatedLowerGiniFeatures                              │
│                                                                 │
│ Output: ~50-100 selected features                              │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: PREPROCESSING (Python - Existing)                     │
├─────────────────────────────────────────────────────────────────┤
│ • WOEBinning (for Layer 1 base model)                         │
│ • OR Target Encoding (for Layer 2 segment models)             │
│ • Scaling (StandardScaler)                                     │
│ • Imputation (SimpleImputer)                                   │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: MODEL TRAINING (Python - Existing)                    │
├─────────────────────────────────────────────────────────────────┤
│ • Layer 1: LogisticRegression + WOE                            │
│ • Layer 2a: LightGBM (Good segment)                           │
│ • Layer 2b: LogisticRegression (Not-Good segment)             │
│ • Layer 3: LogisticRegression (Meta model)                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 New SQL Extraction Query

**File:** `scripts/extract_raw_data.sql` (NEW - replaces scripts 1-6)

```sql
-- Simplified extraction: raw transactions only
WITH loan_data AS (
    SELECT
        -- Identity
        mkr_id,
        (SELECT finreal FROM iibahramova.tmp_mkr_fin_random
         WHERE fin = finrandom) AS fin,
        TO_DATE(mkr_date, 'DD.MM.YYYY') AS mkr_date,

        -- Loan attributes
        credit_type,
        org_type,
        bank_id,
        bank_name,
        credit_purpose,
        currency,

        -- Dates
        granted_on,
        contract_due_on,
        last_update_date,
        last_payment_date,
        credit_status_close_date,

        -- Amounts (keep original currency for conversion in Python)
        initial_amount,
        line_ammount,
        outstanding_debt_main,
        outstanding_debt_interest,
        monthly_payment_amount,
        interest_rate,

        -- Payment behavior
        days_main_sum_overdue,
        days_interest_overdue,
        credit_status AS l_credit_status,
        overdue_days,
        overdue_period,
        prolongations,

        -- Collateral
        collateral_code,
        collateral_market_value,

        -- Identifiers for joining
        id AS person_id,
        id_2 AS loan_id

    FROM scoring.qnb_mkr_data_mba_backup
    WHERE TRUNC(TO_DATE(mkr_date, 'DD.MM.YYYY'))
          BETWEEN TO_DATE('01.01.2018','DD.MM.YYYY')
          AND TO_DATE('31.12.2021','DD.MM.YYYY')
),
salary_data AS (
    SELECT
        fincode AS fin,
        id AS person_id,
        NVL(
            ibs.api_scoring_camunda_main.calc_net_from_gross@ibs_ro(
                salary, l_work_sector_vat
            ),
            salary
        ) AS salary
    FROM (
        SELECT fincode, id,
               SUM(NVL(salary, 0)) AS salary,
               l_work_sector_vat
        FROM (
            SELECT DISTINCT
                b.fincode, b.id,
                a.t_emp_salary AS salary,
                a.l_work_sector_vat
            FROM ibs.asan_finance_is_yeri_v2@ibs_ro a,
                 (SELECT DISTINCT fin AS fincode, id
                  FROM loan_data) b
            WHERE a.t_fin_code = b.fincode
              AND a.t_id = (SELECT MAX(t_id)
                           FROM ibs.asan_finance_is_yeri_v2@ibs_ro k
                           WHERE k.t_fin_code = a.t_fin_code)
              AND a.t_contract_status_desc IS NOT NULL
        )
        GROUP BY fincode, id, l_work_sector_vat
    )
)
SELECT
    l.*,
    s.salary
FROM loan_data l
LEFT JOIN salary_data s
    ON l.fin = s.fin
    AND l.person_id = s.person_id;
```

**Output:**
- ~30 columns instead of 1000+
- Loan-level granularity (~500K rows)
- Raw values for Python transformation

---

## 3. Migration Strategy

### 3.1 Approach: Parallel Run + Gradual Cutover

**Principle:** Don't break existing system while building new one

**Strategy:**
1. ✅ Keep SQL scripts working (baseline)
2. ✅ Build Python feature engineering alongside
3. ✅ Validate Python outputs match SQL outputs
4. ✅ Compare model performance (SQL vs Python features)
5. ✅ Switch to Python when validated
6. ✅ Archive SQL scripts

### 3.2 Feature Group Prioritization

Migrate features in order of:
1. **Complexity** (simple → complex)
2. **Importance** (high Gini → low Gini)
3. **Dependencies** (base features → derived features)

**Migration Order:**

| Priority | Feature Group | Complexity | Lines of SQL | Estimated Python LOC |
|----------|--------------|------------|--------------|---------------------|
| 1 | Currency Conversion | Low | ~50 | ~30 |
| 2 | Product Type Classification | Low | ~20 | ~40 |
| 3 | Count Features (Simple) | Medium | ~500 | ~150 |
| 4 | Amount Aggregations | Medium | ~1000 | ~200 |
| 5 | Payment Status Strings | Medium | ~200 | ~100 |
| 6 | Time-based Aggregations | High | ~2000 | ~300 |
| 7 | Utilization Calculations | High | ~500 | ~150 |
| 8 | Score Codes | Medium | ~50 | ~80 |

**Total:** ~4,320 lines SQL → ~1,050 lines Python (75% reduction)

---

## 4. Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Objectives:**
- Set up new module structure
- Create base classes
- Extract raw data

**Deliverables:**

**1.1 Create new SQL extraction script**
```bash
scripts/extract_raw_data.sql  # New simplified query
```

**1.2 Create feature engineering module**
```bash
QNBAnalytics_ML/feature_engineering.py
```

**1.3 Base transformer class**
```python
# QNBAnalytics_ML/feature_engineering.py

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from QNBAnalytics_ML.helpers import getLogger

class BaseFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for all feature engineering transformers.
    Ensures consistent interface and logging.
    """
    def __init__(self):
        self.logger = getLogger()
        self.feature_names_ = []

    def fit(self, X, y=None):
        """Fit transformer (most will be stateless)"""
        return self

    def transform(self, X):
        """Transform must be implemented by subclass"""
        raise NotImplementedError("Subclass must implement transform()")

    def get_feature_names(self):
        """Return list of feature names created"""
        return self.feature_names_
```

**1.4 Test data extraction**
```python
# tests/test_raw_data_extraction.py

def test_raw_data_structure():
    """Validate raw data has expected columns"""
    df = pd.read_csv('Data/data_raw.csv')

    required_cols = [
        'mkr_id', 'fin', 'mkr_date', 'credit_type',
        'initial_amount', 'currency', 'granted_on',
        'outstanding_debt_main', 'overdue_days', 'salary'
    ]

    assert all(col in df.columns for col in required_cols)
    assert df.shape[0] > 400000  # Expected ~500K loan records
    assert df.shape[1] < 50  # Should be ~30 columns
```

**Success Criteria:**
- ✅ Raw data extracted successfully
- ✅ Base transformer class created
- ✅ Tests pass

---

### Phase 2: Simple Transformers (Week 3-4)

**Objectives:**
- Implement simple, stateless transformers
- Validate outputs match SQL

**Deliverables:**

**2.1 CurrencyConverter**

```python
class CurrencyConverter(BaseFeatureTransformer):
    """
    Converts all monetary amounts to AZN using fixed exchange rates.

    SQL Equivalent: Lines 83-109 in scripts/3.sql

    Rates (as of model training date):
        USD → AZN: 1.7
        EUR → AZN: 1.9
        RUB → AZN: 0.025
        AZN → AZN: 1.0
    """

    def __init__(self,
                 rates={'USD': 1.7, 'EUR': 1.9, 'RUB': 0.025, 'AZN': 1.0}):
        super().__init__()
        self.rates = rates
        self.amount_columns = [
            'initial_amount', 'line_ammount',
            'outstanding_debt_main', 'outstanding_debt_interest',
            'monthly_payment_amount', 'collateral_market_value'
        ]

    def transform(self, X):
        """
        Add _DRV (derived) columns with AZN-converted amounts

        Example:
            initial_amount=1000, currency='USD'
            → initial_amount_DRV=1700
        """
        self.logger.info("Currency conversion started...")
        X = X.copy()

        for col in self.amount_columns:
            if col in X.columns:
                # Create derived column
                derived_col = f"{col}_DRV"
                X[derived_col] = X.apply(
                    lambda row: row[col] * self.rates.get(row['currency'], 1.0)
                    if pd.notna(row[col]) else np.nan,
                    axis=1
                )
                self.feature_names_.append(derived_col)

        # Total outstanding balance (main + interest)
        X['total_outstanding_balance'] = (
            X['outstanding_debt_main_DRV'] +
            X['outstanding_debt_interest_DRV']
        )
        self.feature_names_.append('total_outstanding_balance')

        self.logger.info(f"Currency conversion completed. Created {len(self.feature_names_)} features.")
        return X
```

**2.2 ProductTypeClassifier**

```python
class ProductTypeClassifier(BaseFeatureTransformer):
    """
    Classifies loans into product categories.

    SQL Equivalent: Lines 60-64 in scripts/3.sql

    Product Types:
        CL  = Consumer Loan (001)
        CC  = Credit Card (003)
        HL  = Housing Loan (300 if purpose=003, else 001)
        OL  = Other Loan (002, 005)
        CHL = Consumer + Housing (combined)
        CCOL = Credit Card + Other Loan (combined)
        ALL = All products
    """

    def transform(self, X):
        self.logger.info("Product classification started...")
        X = X.copy()

        # Derived credit type
        X['credit_type_drv'] = X.apply(self._classify_product, axis=1)

        # Product category flags
        X['is_CL'] = (X['credit_type_drv'] == '001').astype(int)
        X['is_CC'] = (X['credit_type_drv'] == '003').astype(int)
        X['is_HL'] = (X['credit_type_drv'] == '300').astype(int)
        X['is_OL'] = (X['credit_type_drv'] == '002').astype(int)
        X['is_CHL'] = X[['is_CL', 'is_HL']].max(axis=1)
        X['is_CCOL'] = X[['is_CC', 'is_OL']].max(axis=1)
        X['is_ALL'] = 1  # All loans

        self.feature_names_ = [
            'credit_type_drv', 'is_CL', 'is_CC', 'is_HL', 'is_OL',
            'is_CHL', 'is_CCOL', 'is_ALL'
        ]

        self.logger.info("Product classification completed.")
        return X

    def _classify_product(self, row):
        """Business logic for product classification"""
        if row['credit_type'] == '003':
            return '003'  # Credit Card
        elif row['credit_type'] in ['002', '005']:
            return '002'  # Other Loan
        elif row['credit_type'] == '001' and row['credit_purpose'] == '003':
            return '300'  # Housing Loan
        else:
            return '001'  # Consumer Loan
```

**2.3 Validation Tests**

```python
# tests/test_simple_transformers.py

def test_currency_converter():
    """Validate currency conversion matches SQL output"""
    # Load test data
    df_raw = pd.read_csv('Data/data_raw.csv', nrows=1000)
    df_sql = pd.read_csv('Data/data.csv', nrows=1000)

    # Apply transformer
    converter = CurrencyConverter()
    df_python = converter.transform(df_raw)

    # Compare: Python vs SQL
    for col in ['initial_amount_DRV', 'outstanding_debt_main_DRV']:
        if col in df_sql.columns:
            np.testing.assert_array_almost_equal(
                df_python[col].fillna(0),
                df_sql[col].fillna(0),
                decimal=2,
                err_msg=f"{col} mismatch between Python and SQL"
            )

def test_product_classifier():
    """Validate product classification logic"""
    df_raw = pd.read_csv('Data/data_raw.csv', nrows=1000)
    df_sql = pd.read_csv('Data/data.csv', nrows=1000)

    classifier = ProductTypeClassifier()
    df_python = classifier.transform(df_raw)

    # Validate classification logic
    assert df_python['credit_type_drv'].isin(['001', '002', '003', '300']).all()

    # Compare counts with SQL
    if 'credit_type_drv' in df_sql.columns:
        assert (df_python['credit_type_drv'] == df_sql['credit_type_drv']).sum() > 900
```

**Success Criteria:**
- ✅ Currency conversion: Python == SQL (within 0.01 precision)
- ✅ Product classification: 100% match with SQL
- ✅ Tests pass

---

### Phase 3: Aggregation Transformers (Week 5-8)

**Objectives:**
- Implement complex aggregation logic
- Create count and amount features
- Handle time-based windows

**Deliverables:**

**3.1 CreditBureauAggregator**

```python
class CreditBureauAggregator(BaseFeatureTransformer):
    """
    Aggregates loan-level data to person-level features.

    SQL Equivalent: Lines 193-4069 in scripts/3.sql

    Creates features following pattern:
        {PRODUCT}_{METRIC}_{TIME}_{FILTER}_{STATUS}

    Examples:
        CL_OCNT_90D                      # Consumer loans opened in 90 days
        CC_OSMTOB_EVER_O                 # CC sum total balance, ever, open only
        ALL_OCNT_CWPS0_182D              # All accounts, no delinquency, 182 days
        HL_OLMTUTL_CWPS0_365D_O          # HL utilization, current, 365d, open
    """

    def __init__(self,
                 products=['CL', 'CC', 'HL', 'OL', 'CHL', 'CCOL', 'ALL'],
                 time_periods=[30, 90, 182, 365],
                 metrics=['count', 'sum_amount', 'avg_amount', 'max_amount',
                         'sum_balance', 'avg_balance', 'utilization']):
        super().__init__()
        self.products = products
        self.time_periods = time_periods + ['EVER']  # Add "all time"
        self.metrics = metrics

    def transform(self, X):
        """
        Main aggregation logic.

        Input: Loan-level dataframe (500K rows)
        Output: Person-level dataframe with aggregated features (96K rows)
        """
        self.logger.info("Credit bureau aggregation started...")
        X = X.copy()

        # Ensure datetime
        X['mkr_date'] = pd.to_datetime(X['mkr_date'])
        X['granted_on'] = pd.to_datetime(X['granted_on'])

        # Calculate days since granted
        X['days_since_granted'] = (X['mkr_date'] - X['granted_on']).dt.days

        # Determine account status (Open/Closed)
        X['is_open'] = self._determine_account_status(X)

        # Determine payment status category
        X['payment_status_cat'] = self._categorize_payment_status(X)

        # Group by person and aggregate
        grouped = X.groupby(['fin', 'mkr_date'])

        aggregated_features = []

        for product in self.products:
            for time_period in self.time_periods:
                for status_filter in [None, 'O']:  # None=All, O=Open only
                    # Filter data
                    df_filtered = self._filter_data(
                        X, product, time_period, status_filter
                    )

                    # Aggregate
                    agg_df = self._aggregate_features(
                        df_filtered, product, time_period, status_filter
                    )

                    aggregated_features.append(agg_df)

        # Combine all aggregations
        result = aggregated_features[0]
        for agg_df in aggregated_features[1:]:
            result = result.merge(agg_df, on=['fin', 'mkr_date'], how='outer')

        self.logger.info(f"Aggregation completed. Created {len(result.columns)-2} features.")
        self.feature_names_ = [c for c in result.columns if c not in ['fin', 'mkr_date']]

        return result

    def _filter_data(self, X, product, time_period, status_filter):
        """Apply product, time, and status filters"""
        df = X.copy()

        # Product filter
        if product != 'ALL':
            df = df[df[f'is_{product}'] == 1]

        # Time period filter
        if time_period != 'EVER':
            df = df[df['days_since_granted'] <= time_period]

        # Account status filter (Open only)
        if status_filter == 'O':
            df = df[df['is_open'] == 1]

        return df

    def _aggregate_features(self, df, product, time_period, status_filter):
        """
        Calculate aggregations for filtered data

        Returns DataFrame with columns like:
            CL_OCNT_90D, CL_OSMAMT_90D, CL_OMXAMT_90D, etc.
        """
        suffix = f"{time_period}" if time_period != 'EVER' else 'EVER'
        if status_filter == 'O':
            suffix += '_O'

        agg_dict = {
            # Count features
            f'{product}_OCNT_{suffix}': ('loan_id', 'count'),

            # Amount features (initial amounts)
            f'{product}_OSMAMT_{suffix}': ('initial_amount_DRV', 'sum'),
            f'{product}_OAVGAMT_{suffix}': ('initial_amount_DRV', 'mean'),
            f'{product}_OMXAMT_{suffix}': ('initial_amount_DRV', 'max'),

            # Balance features (outstanding balances)
            f'{product}_OSMTOB_{suffix}': ('total_outstanding_balance', 'sum'),
            f'{product}_OAVGTOB_{suffix}': ('total_outstanding_balance', 'mean'),
            f'{product}_OMXTOB_{suffix}': ('total_outstanding_balance', 'max'),

            # Limit features (credit cards / lines)
            f'{product}_OSMLMT_{suffix}': ('line_ammount_DRV', 'sum'),
        }

        result = df.groupby(['fin', 'mkr_date']).agg(**agg_dict).reset_index()

        return result

    def _determine_account_status(self, X):
        """
        Determine if account is open or closed.
        SQL Equivalent: Lines 111-116 in scripts/3.sql
        """
        is_open = (
            (X['l_credit_status'].isin(['001', '010', '012'])) |  # Active statuses
            (X['total_outstanding_balance'] > 0)  # Has balance
        ).astype(int)

        return is_open

    def _categorize_payment_status(self, X):
        """
        Categorize payment status:
            - CWPS0: Current (no delinquency)
            - CWPS1_6: Minor delinquency (1-6 months)
            - CWPS6P: Major delinquency (6+ months)
            - CWPSUD: Undefined
        """
        conditions = [
            X['overdue_days'] == 0,
            X['overdue_days'].between(1, 186),
            X['overdue_days'] > 186,
        ]
        categories = ['CWPS0', 'CWPS1_6', 'CWPS6P']

        return np.select(conditions, categories, default='CWPSUD')
```

**3.2 PaymentStatusEncoder**

```python
class PaymentStatusEncoder(BaseFeatureTransformer):
    """
    Creates historical payment status strings.

    SQL Equivalent: Lines 129-185 in scripts/3.sql

    Output format: String of digits representing monthly payment history
    Example: "00012390XX"
        0 = Current
        1-9 = Months overdue (1=1-31 days, 2=32-62, ..., 9=279+)
        X = Unknown/missing

    Features created:
        ACCOUNT_PAYMENT_STATUS: Historical string for each loan
        Plus aggregated features like ALL_CWPS_EVER_N (count of worst statuses)
    """

    def __init__(self, lookback_months=24):
        super().__init__()
        self.lookback_months = lookback_months

    def transform(self, X):
        """Generate payment status strings"""
        self.logger.info("Payment status encoding started...")
        X = X.copy()

        # This requires overdue_period and overdue_days columns
        # Group by loan and create historical string

        def encode_overdue(days):
            """Convert days overdue to status code"""
            if pd.isna(days):
                return 'X'
            elif days == 0:
                return '0'
            elif days >= 279:
                return '9'
            else:
                return str(min(int(np.ceil(days / 31)), 9))

        # Apply encoding
        X['payment_status_code'] = X['overdue_days'].apply(encode_overdue)

        # Aggregate to person level (concatenate monthly history)
        # This is complex - need to pivot by month and concatenate
        # Simplified version for now:
        payment_history = X.groupby(['fin', 'mkr_date', 'loan_id']).agg({
            'payment_status_code': lambda x: ''.join(x.astype(str))
        }).reset_index()

        payment_history.rename(columns={
            'payment_status_code': 'ACCOUNT_PAYMENT_STATUS'
        }, inplace=True)

        self.logger.info("Payment status encoding completed.")
        self.feature_names_ = ['ACCOUNT_PAYMENT_STATUS']

        return payment_history
```

**3.3 Validation Strategy**

```python
# tests/test_aggregation.py

def test_count_features():
    """Validate count features match SQL exactly"""
    df_raw = pd.read_csv('Data/data_raw.csv')
    df_sql = pd.read_csv('Data/data.csv')

    # Apply transformations
    aggregator = CreditBureauAggregator()
    df_python = aggregator.transform(df_raw)

    # Test specific features
    test_features = [
        'CL_OCNT_90D',
        'CC_OCNT_EVER',
        'HL_OCNT_182D_O',
        'ALL_OCNT_CWPS0_365D'
    ]

    for feat in test_features:
        if feat in df_sql.columns:
            # Merge on person+date
            comparison = df_python.merge(
                df_sql[['fin', 'mkr_date', feat]],
                on=['fin', 'mkr_date'],
                suffixes=('_py', '_sql')
            )

            # Check match rate
            match_rate = (comparison[f'{feat}_py'] == comparison[f'{feat}_sql']).mean()
            assert match_rate > 0.99, f"{feat}: Only {match_rate*100:.1f}% match"

def test_amount_features():
    """Validate amount aggregations"""
    # Similar to count features but allow small rounding differences
    pass
```

**Success Criteria:**
- ✅ Count features: >99% exact match with SQL
- ✅ Amount features: >99% match within 0.01 AZN
- ✅ No memory errors on full dataset
- ✅ Performance: <10 minutes on 500K rows

---

### Phase 4: Advanced Features (Week 9-10)

**Objectives:**
- Implement utilization calculations
- Implement score codes
- Complete all feature engineering

**Deliverables:**

**4.1 UtilizationCalculator**

```python
class UtilizationCalculator(BaseFeatureTransformer):
    """
    Calculates credit utilization ratios.

    SQL Equivalent: Lines 5100-5119 in scripts/3.sql

    Utilization = Outstanding Balance / Credit Limit

    Example features:
        CC_OLMTUTL_EVER_O    # Credit card utilization, all time, open
        CL_OAVGINSDIVSMTOB_90D  # CL avg installment / sum balance, 90d
    """

    def transform(self, X):
        """Calculate utilization ratios"""
        self.logger.info("Utilization calculation started...")
        X = X.copy()

        # Need aggregated features as input
        # Calculate ratios: balance / limit

        products = ['CL', 'CC', 'HL', 'OL', 'CHL', 'CCOL', 'ALL']
        time_periods = ['EVER', '90D', '182D', '183D365D', '365D']
        statuses = ['', '_O']  # All accounts, Open only

        for product in products:
            for period in time_periods:
                for status in statuses:
                    suffix = period + status

                    # Utilization = balance / limit
                    lmt_col = f'{product}_OSMLMT_{suffix}'
                    tob_col = f'{product}_OSMTOB_{suffix}'
                    util_col = f'{product}_OLMTUTL_{suffix}'

                    if lmt_col in X.columns and tob_col in X.columns:
                        X[util_col] = (
                            X[tob_col] / X[lmt_col]
                        ).replace([np.inf, -np.inf], np.nan).round(6)

                        self.feature_names_.append(util_col)

        self.logger.info(f"Utilization calculation completed. Created {len(self.feature_names_)} features.")
        return X
```

**4.2 ScoreCodeCalculator**

```python
class ScoreCodeCalculator(BaseFeatureTransformer):
    """
    Calculates score adjustment codes.

    SQL Equivalent: Lines 5124-5148 in scripts/3.sql

    Score codes penalize/reward based on delinquency severity:
        SCR_CONSTANT: Additive adjustment (-200, -100, +140)
        SCR_MULTIPLIER: Multiplicative adjustment (0.3-0.7)
    """

    def transform(self, X):
        """Calculate score codes"""
        self.logger.info("Score code calculation started...")
        X = X.copy()

        # SC1: -200 if >3 current worst payment statuses
        X['SC1'] = np.where(X['ALL_CWPS_EVER_N'] > 3, -200, np.nan)

        # SC2: -100 if 2-3 current worst payment statuses
        X['SC2'] = np.where(
            X['ALL_CWPS_EVER_N'].between(2, 3), -100, np.nan
        )

        # SC3-SC6: Multipliers based on historical delinquency
        X['SC3'] = np.where(
            (X['ALL_2M3MWPS_EVER_N'] > 3) &
            (X['ALL_SMTOB_2M3MWPS4P_EVER_N'] / X['CHL_OSMAMT_EVER'] >= 0.05),
            0.3, np.nan
        )

        # ... (similar logic for SC4, SC5, SC6)

        # SC8: +140 for recovered minor delinquency
        X['SC8'] = np.where(
            (X['ALL_CWPS_EVER_N'] == 1) &
            X['MX_LAST_OVERDUE_DAYS'].between(6, 31),
            140, np.nan
        )

        # Final codes
        X['SCR_CONSTANT'] = X[['SC1', 'SC2', 'SC8']].min(axis=1, skipna=False).fillna(0)
        X['SCR_MULTIPLIER'] = X[['SC3', 'SC4', 'SC5', 'SC6']].min(axis=1, skipna=False).fillna(1.0)

        self.feature_names_ = ['SCR_CONSTANT', 'SCR_MULTIPLIER']
        self.logger.info("Score code calculation completed.")
        return X
```

**4.3 Main Pipeline Orchestrator**

```python
class FeatureEngineeringPipeline(BaseFeatureTransformer):
    """
    Main orchestrator for all feature engineering steps.

    Usage:
        pipeline = FeatureEngineeringPipeline()
        df_features = pipeline.fit_transform(df_raw)
    """

    def __init__(self):
        super().__init__()

        # Define pipeline steps
        self.steps = [
            ('currency', CurrencyConverter()),
            ('product', ProductTypeClassifier()),
            ('aggregator', CreditBureauAggregator()),
            ('payment_status', PaymentStatusEncoder()),
            ('utilization', UtilizationCalculator()),
            ('score_codes', ScoreCodeCalculator()),
        ]

    def fit(self, X, y=None):
        """Fit all transformers"""
        for name, transformer in self.steps:
            transformer.fit(X, y)
        return self

    def transform(self, X):
        """Apply all transformations sequentially"""
        self.logger.info("Feature engineering pipeline started...")

        X_transformed = X.copy()

        for name, transformer in self.steps:
            self.logger.info(f"  Step: {name}")
            X_transformed = transformer.transform(X_transformed)

        # Collect all feature names
        self.feature_names_ = []
        for name, transformer in self.steps:
            self.feature_names_.extend(transformer.get_feature_names())

        self.logger.info(f"Feature engineering completed. Total features: {len(self.feature_names_)}")
        return X_transformed

    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
```

**Success Criteria:**
- ✅ All 1000+ features generated
- ✅ >95% match with SQL features
- ✅ Pipeline runs end-to-end
- ✅ Memory efficient (<16GB RAM)

---

### Phase 5: Integration with Existing Pipeline (Week 11)

**Objectives:**
- Connect new feature engineering with existing elimination/binning
- Update training notebook
- Test complete pipeline

**Deliverables:**

**5.1 Updated Training Script**

```python
# BOB_Scorecard_Training_v2.ipynb (new version)

import pandas as pd
from sklearn.model_selection import train_test_split

# Import new feature engineering
from QNBAnalytics_ML.feature_engineering import FeatureEngineeringPipeline

# Import existing modules
from QNBAnalytics_ML.elimination import (
    DropNullFeatures, DropConstantFeatures,
    DropLowGiniFeatures, DropCorrelatedLowerGiniFeatures
)
from QNBAnalytics_ML.binning import WOEBinning
from QNBAnalytics_ML.preprocessing import scale, impute_null, encode
from QNBAnalytics_ML.classifier import LogisticRegression, LightGBM

# ========================================
# STEP 1: Load raw data
# ========================================
print("Loading raw data...")
data_raw = pd.read_csv('Data/data_raw.csv')
print(f"Raw data shape: {data_raw.shape}")

# ========================================
# STEP 2: Feature Engineering (NEW)
# ========================================
print("\n=== Feature Engineering ===")
fe_pipeline = FeatureEngineeringPipeline()
data_features = fe_pipeline.fit_transform(data_raw)
print(f"Features created: {data_features.shape[1]}")

# Save for comparison
data_features.to_csv('Data/data_features_python.csv', index=False)

# ========================================
# STEP 3: Train/Test Split (Temporal)
# ========================================
print("\n=== Temporal Split ===")
# Sort by date
data_features = data_features.sort_values('mkr_date')

# Split: 70% train, 30% test (temporal)
split_idx = int(len(data_features) * 0.7)
train_data = data_features.iloc[:split_idx]
test_data = data_features.iloc[split_idx:]

print(f"Train: {train_data.shape}, Test: {test_data.shape}")

# Separate X and y
feature_cols = [c for c in train_data.columns
                if c not in ['TARGET', 'MUQAVILE', 'FIN', 'MKR_DATE']]
X_train = train_data[feature_cols]
y_train = train_data['TARGET']
X_test = test_data[feature_cols]
y_test = test_data['TARGET']

# ========================================
# STEP 4: Feature Elimination (EXISTING)
# ========================================
print("\n=== Feature Elimination ===")

# Drop null features
drop_null = DropNullFeatures(threshold=0.99)
drop_null.fit(X_train, y_train)
X_train = drop_null.transform(X_train)
X_test = drop_null.transform(X_test)
print(f"After null elimination: {X_train.shape[1]} features")

# Drop constant features
drop_constant = DropConstantFeatures()
drop_constant.fit(X_train, y_train)
X_train = drop_constant.transform(X_train)
X_test = drop_constant.transform(X_test)
print(f"After constant elimination: {X_train.shape[1]} features")

# Drop low Gini features
drop_low_gini = DropLowGiniFeatures(threshold=0.05)
drop_low_gini.fit(X_train, y_train)
X_train = drop_low_gini.transform(X_train)
X_test = drop_low_gini.transform(X_test)
print(f"After Gini elimination: {X_train.shape[1]} features")

# Drop correlated features
drop_corr = DropCorrelatedLowerGiniFeatures(threshold=0.9)
drop_corr.fit(X_train, y_train)
X_train = drop_corr.transform(X_train)
X_test = drop_corr.transform(X_test)
print(f"After correlation elimination: {X_train.shape[1]} features")

# ========================================
# STEP 5: WOE Binning (EXISTING - Layer 1)
# ========================================
print("\n=== WOE Binning (Layer 1 Base Model) ===")
woe = WOEBinning()
woe.fit(X_train, y_train)
X_train_woe = woe.transform(X_train, y_train)
X_test_woe = woe.transform(X_test)

# ========================================
# STEP 6: Train Layer 1 Base Model
# ========================================
print("\n=== Training Layer 1: Base Model (LR + WOE) ===")
base_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    max_iter=1000,
    random_state=42
)
base_model.fit(X_train_woe, y_train)

# Predictions
train_proba_base = base_model.predict_proba(X_train_woe)[:, 1]
test_proba_base = base_model.predict_proba(X_test_woe)[:, 1]

# Evaluate
from sklearn.metrics import roc_auc_score
train_gini_base = 2 * roc_auc_score(y_train, train_proba_base) - 1
test_gini_base = 2 * roc_auc_score(y_test, test_proba_base) - 1

print(f"Layer 1 Train Gini: {train_gini_base:.4f}")
print(f"Layer 1 Test Gini: {test_gini_base:.4f}")

# ========================================
# STEP 7-9: Layer 2 and Layer 3 (EXISTING)
# ========================================
# ... (keep existing logic for segment models and meta model)
```

**5.2 Comparison Script**

```python
# scripts/compare_sql_vs_python.py

"""
Compares feature outputs from SQL vs Python pipelines.
Validates that migration was successful.
"""

import pandas as pd
import numpy as np

def compare_features(sql_file, python_file, tolerance=0.01):
    """
    Compare SQL and Python feature outputs.

    Args:
        sql_file: Path to CSV from SQL pipeline (data.csv)
        python_file: Path to CSV from Python pipeline (data_features_python.csv)
        tolerance: Maximum allowed difference for numerical features

    Returns:
        DataFrame with comparison results
    """
    print("Loading data...")
    df_sql = pd.read_csv(sql_file)
    df_py = pd.read_csv(python_file)

    print(f"SQL features: {df_sql.shape[1]}")
    print(f"Python features: {df_py.shape[1]}")

    # Merge on keys
    merged = df_sql.merge(
        df_py,
        on=['MUQAVILE', 'FIN', 'MKR_DATE'],
        suffixes=('_sql', '_py'),
        how='inner'
    )

    print(f"Merged rows: {len(merged)}")

    # Find common features
    sql_features = [c for c in df_sql.columns if c not in ['MUQAVILE', 'FIN', 'MKR_DATE']]
    py_features = [c for c in df_py.columns if c not in ['MUQAVILE', 'FIN', 'MKR_DATE']]

    common_features = set(sql_features) & set(py_features)
    print(f"Common features: {len(common_features)}")

    # Compare each feature
    results = []

    for feat in common_features:
        sql_col = f"{feat}_sql"
        py_col = f"{feat}_py"

        if sql_col not in merged.columns or py_col not in merged.columns:
            continue

        # Calculate match rate
        sql_vals = merged[sql_col].fillna(-999)
        py_vals = merged[py_col].fillna(-999)

        # Exact match
        exact_match = (sql_vals == py_vals).mean()

        # Close match (within tolerance)
        close_match = (np.abs(sql_vals - py_vals) < tolerance).mean()

        # Max difference
        max_diff = np.abs(sql_vals - py_vals).max()

        results.append({
            'feature': feat,
            'exact_match_pct': exact_match * 100,
            'close_match_pct': close_match * 100,
            'max_diff': max_diff,
            'sql_mean': merged[sql_col].mean(),
            'py_mean': merged[py_col].mean(),
            'status': 'PASS' if close_match > 0.95 else 'FAIL'
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('close_match_pct')

    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Total features compared: {len(results_df)}")
    print(f"PASS (>95% match): {(results_df['status']=='PASS').sum()}")
    print(f"FAIL (<95% match): {(results_df['status']=='FAIL').sum()}")
    print(f"\nAverage exact match: {results_df['exact_match_pct'].mean():.2f}%")
    print(f"Average close match: {results_df['close_match_pct'].mean():.2f}%")

    # Show failures
    failures = results_df[results_df['status'] == 'FAIL']
    if len(failures) > 0:
        print("\n" + "="*60)
        print("FAILED FEATURES (need investigation):")
        print("="*60)
        print(failures[['feature', 'close_match_pct', 'max_diff']].to_string())

    return results_df

if __name__ == '__main__':
    results = compare_features(
        'Data/data.csv',
        'Data/data_features_python.csv',
        tolerance=0.01
    )

    results.to_csv('Output/feature_comparison.csv', index=False)
    print("\nResults saved to Output/feature_comparison.csv")
```

**Success Criteria:**
- ✅ >95% of features match between SQL and Python
- ✅ Model performance within 1% (Gini difference <0.01)
- ✅ Training notebook runs end-to-end
- ✅ No breaking changes to existing classes

---

### Phase 6: Production Readiness (Week 12)

**Objectives:**
- Optimize performance
- Add production scoring capability
- Documentation and handoff

**Deliverables:**

**6.1 Production Scoring Script**

```python
# scripts/score_new_applicant.py

"""
Score a new loan applicant using the complete Python pipeline.
"""

import pandas as pd
import joblib
from QNBAnalytics_ML.feature_engineering import FeatureEngineeringPipeline

def score_applicant(fin_code, mkr_date):
    """
    Score a new applicant.

    Args:
        fin_code: Applicant tax ID
        mkr_date: Credit bureau report date

    Returns:
        dict with score and probability
    """
    # Step 1: Fetch raw credit history (from database or API)
    raw_data = fetch_credit_history(fin_code, mkr_date)

    # Step 2: Feature engineering
    fe_pipeline = joblib.load('Models/feature_engineering_pipeline.pkl')
    features = fe_pipeline.transform(raw_data)

    # Step 3: Feature elimination
    elimination_pipeline = joblib.load('Models/elimination_pipeline.pkl')
    features = elimination_pipeline.transform(features)

    # Step 4: WOE binning
    woe = joblib.load('Models/woe_binning.pkl')
    features_woe = woe.transform(features)

    # Step 5: Predict
    model = joblib.load('Models/base_model_training.pkl')
    proba = model.predict_proba(features_woe)[0, 1]

    # Step 6: Convert to score
    score = probability_to_score(proba, REF=200, PDO=20, ODDS_REF=100)

    # Step 7: Apply policy adjustment
    score_final = min(score, 250) * 0.95

    return {
        'fin_code': fin_code,
        'probability': proba,
        'raw_score': score,
        'final_score': score_final,
        'risk_category': categorize_risk(score_final)
    }

def fetch_credit_history(fin_code, mkr_date):
    """Fetch raw credit history from database"""
    # SQL query to get loan-level data
    query = f"""
    SELECT * FROM mkr_data
    WHERE fin = '{fin_code}'
    AND mkr_date = TO_DATE('{mkr_date}', 'YYYY-MM-DD')
    """
    # Execute and return DataFrame
    pass

def probability_to_score(prob, REF=200, PDO=20, ODDS_REF=100):
    """Convert probability to credit score"""
    import numpy as np
    odds = (1 - prob) / prob
    score = ((np.log(odds) - np.log(ODDS_REF)) / np.log(2)) * PDO + REF
    return score

def categorize_risk(score):
    """Categorize score into risk bands"""
    if score >= 220:
        return 'Excellent'
    elif score >= 200:
        return 'Good'
    elif score >= 180:
        return 'Fair'
    else:
        return 'Poor'

# Example usage
if __name__ == '__main__':
    result = score_applicant(
        fin_code='ABC123XYZ',  # Anonymized
        mkr_date='2024-01-15'
    )
    print(f"Final Score: {result['final_score']:.0f}")
    print(f"Risk Category: {result['risk_category']}")
    print(f"Default Probability: {result['probability']:.2%}")
```

**6.2 Performance Optimization**

```python
# QNBAnalytics_ML/feature_engineering.py (optimized)

class CreditBureauAggregator(BaseFeatureTransformer):
    """
    Optimized version using vectorized operations.
    """

    def transform(self, X):
        """Use groupby.agg() with multiple functions at once"""
        # Instead of looping, use pivot tables and vectorized ops

        # Vectorized time filtering
        X['period_30D'] = (X['days_since_granted'] <= 30).astype(int)
        X['period_90D'] = (X['days_since_granted'] <= 90).astype(int)
        X['period_182D'] = (X['days_since_granted'] <= 182).astype(int)
        # ... etc

        # Single groupby with all aggregations
        agg_dict = {}
        for product in self.products:
            for period in self.time_periods:
                # Build dict for pd.NamedAgg
                agg_dict.update(self._build_agg_dict(product, period))

        # One groupby instead of many
        result = X.groupby(['fin', 'mkr_date']).agg(**agg_dict).reset_index()

        return result
```

**6.3 Documentation**

- ✅ Create `docs/PYTHON_FEATURE_ENGINEERING_GUIDE.md`
- ✅ Update `README.md` with new pipeline instructions
- ✅ Add docstrings to all classes and methods
- ✅ Create example notebooks in `examples/`

**Success Criteria:**
- ✅ Production scoring works end-to-end
- ✅ Scoring time: <5 seconds per applicant
- ✅ Memory usage: <4GB for single applicant
- ✅ Documentation complete

---

## 5. Integration with Existing Code

### 5.1 No Changes Required

The following existing modules require **NO CHANGES**:

- ✅ `QNBAnalytics_ML/elimination.py` - Feature selection works on any features
- ✅ `QNBAnalytics_ML/binning.py` - WOE binning agnostic to feature source
- ✅ `QNBAnalytics_ML/preprocessing.py` - Scaling/encoding unchanged
- ✅ `QNBAnalytics_ML/classifier.py` - Models unchanged
- ✅ `QNBAnalytics_ML/skills_api.py` - API unchanged
- ✅ `QNBAnalytics_ML/kpi.py` - Evaluation metrics unchanged

### 5.2 New Files Added

```
QNBAnalytics_ML/
├── feature_engineering.py          # ← NEW (main module)
└── __init__.py                      # ← UPDATE (add new imports)

scripts/
├── extract_raw_data.sql            # ← NEW (simplified extraction)
├── compare_sql_vs_python.py        # ← NEW (validation)
└── score_new_applicant.py          # ← NEW (production scoring)

tests/
├── test_feature_engineering.py     # ← NEW (unit tests)
├── test_simple_transformers.py     # ← NEW
├── test_aggregation.py             # ← NEW
└── test_end_to_end.py              # ← NEW (integration tests)

docs/
├── FEATURE_ENGINEERING_MIGRATION_PLAN.md  # ← THIS DOCUMENT
└── PYTHON_FEATURE_ENGINEERING_GUIDE.md    # ← NEW (user guide)

examples/
└── feature_engineering_demo.ipynb  # ← NEW (tutorial)
```

### 5.3 Updated Files

```
BOB_Scorecard_Training.ipynb        # ← UPDATE (use new pipeline)
requirements.txt                     # ← UPDATE (add dependencies)
README.md                            # ← UPDATE (new instructions)
```

### 5.4 Deprecated Files (Archive Only)

```
scripts/1.sql → scripts/archive/1.sql  # Keep for reference
scripts/2.sql → scripts/archive/2.sql
scripts/3.sql → scripts/archive/3.sql  # 5,148 lines - no longer used!
scripts/4.sql → scripts/archive/4.sql
scripts/5.sql → scripts/archive/5.sql
scripts/6.sql → scripts/archive/6.sql
```

---

## 6. Testing & Validation

### 6.1 Unit Tests

Test each transformer independently:

```python
# tests/test_feature_engineering.py

import pytest
import pandas as pd
import numpy as np
from QNBAnalytics_ML.feature_engineering import (
    CurrencyConverter, ProductTypeClassifier,
    CreditBureauAggregator, PaymentStatusEncoder
)

class TestCurrencyConverter:
    def test_usd_conversion(self):
        df = pd.DataFrame({
            'initial_amount': [1000],
            'currency': ['USD']
        })
        converter = CurrencyConverter()
        result = converter.transform(df)
        assert result['initial_amount_DRV'].iloc[0] == 1700

    def test_eur_conversion(self):
        df = pd.DataFrame({
            'initial_amount': [1000],
            'currency': ['EUR']
        })
        converter = CurrencyConverter()
        result = converter.transform(df)
        assert result['initial_amount_DRV'].iloc[0] == 1900

    def test_azn_passthrough(self):
        df = pd.DataFrame({
            'initial_amount': [1000],
            'currency': ['AZN']
        })
        converter = CurrencyConverter()
        result = converter.transform(df)
        assert result['initial_amount_DRV'].iloc[0] == 1000

class TestProductTypeClassifier:
    def test_credit_card_classification(self):
        df = pd.DataFrame({
            'credit_type': ['003'],
            'credit_purpose': ['001']
        })
        classifier = ProductTypeClassifier()
        result = classifier.transform(df)
        assert result['credit_type_drv'].iloc[0] == '003'
        assert result['is_CC'].iloc[0] == 1

    def test_housing_loan_classification(self):
        df = pd.DataFrame({
            'credit_type': ['001'],
            'credit_purpose': ['003']
        })
        classifier = ProductTypeClassifier()
        result = classifier.transform(df)
        assert result['credit_type_drv'].iloc[0] == '300'
        assert result['is_HL'].iloc[0] == 1

# ... more unit tests
```

### 6.2 Integration Tests

Test full pipeline:

```python
# tests/test_end_to_end.py

def test_full_pipeline():
    """Test complete pipeline from raw data to model predictions"""

    # Load raw data
    df_raw = pd.read_csv('Data/data_raw.csv', nrows=10000)

    # Feature engineering
    fe_pipeline = FeatureEngineeringPipeline()
    df_features = fe_pipeline.fit_transform(df_raw)

    # Should create 1000+ features
    assert df_features.shape[1] > 1000

    # Should not have NaN in key columns
    assert df_features['SCR_CONSTANT'].notna().sum() > 0

    # Feature elimination
    drop_null = DropNullFeatures()
    drop_null.fit(df_features, df_raw['TARGET'])
    df_selected = drop_null.transform(df_features)

    # WOE binning
    woe = WOEBinning()
    woe.fit(df_selected, df_raw['TARGET'])
    df_woe = woe.transform(df_selected)

    # Model training
    model = LogisticRegression()
    model.fit(df_woe, df_raw['TARGET'])

    # Predictions
    proba = model.predict_proba(df_woe)[:, 1]

    # Should produce valid probabilities
    assert (proba >= 0).all() and (proba <= 1).all()
```

### 6.3 Validation Metrics

Track these metrics during migration:

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Feature match rate | >95% | Compare SQL vs Python outputs |
| Model Gini (Layer 1 Test) | ≥0.43 | Train on Python features |
| Model Gini (Layer 3 Test) | ≥0.42 | Full 3-layer ensemble |
| Feature count | ~1000 | Count engineered features |
| Processing time | <10 min | Time full pipeline on 500K rows |
| Memory usage | <16GB | Monitor peak RAM usage |
| Scoring time (single) | <5 sec | Time production scoring |

### 6.4 Regression Testing

Ensure no degradation:

```python
# tests/test_regression.py

def test_no_model_degradation():
    """Ensure Python pipeline doesn't degrade model performance"""

    # Baseline: SQL features
    df_sql = pd.read_csv('Data/data.csv')
    X_sql = df_sql.drop(['TARGET'], axis=1)
    y_sql = df_sql['TARGET']

    X_train_sql, X_test_sql, y_train, y_test = train_test_split(
        X_sql, y_sql, test_size=0.3, random_state=42
    )

    model_sql = LogisticRegression()
    model_sql.fit(X_train_sql, y_train)
    gini_sql = 2 * roc_auc_score(y_test, model_sql.predict_proba(X_test_sql)[:,1]) - 1

    # New: Python features
    df_raw = pd.read_csv('Data/data_raw.csv')
    fe = FeatureEngineeringPipeline()
    df_py = fe.fit_transform(df_raw)

    X_py = df_py.drop(['TARGET'], axis=1)
    y_py = df_py['TARGET']

    X_train_py, X_test_py, y_train, y_test = train_test_split(
        X_py, y_py, test_size=0.3, random_state=42
    )

    model_py = LogisticRegression()
    model_py.fit(X_train_py, y_train)
    gini_py = 2 * roc_auc_score(y_test, model_py.predict_proba(X_test_py)[:,1]) - 1

    # Python should match or exceed SQL performance
    assert gini_py >= gini_sql - 0.01  # Allow 1% tolerance
```

---

## 7. Deployment Plan

### 7.1 Rollout Strategy

**Phase 1: Shadow Mode (Week 13-14)**
- Run Python pipeline in parallel with SQL
- Compare outputs daily
- No production usage yet

**Phase 2: A/B Testing (Week 15-16)**
- Score 10% of new applications with Python pipeline
- Compare scores with SQL pipeline
- Monitor for discrepancies

**Phase 3: Full Cutover (Week 17)**
- Switch 100% of production traffic to Python pipeline
- Archive SQL scripts
- Monitor performance for 1 week

**Phase 4: Optimization (Week 18+)**
- Profile and optimize bottlenecks
- Add caching for frequently-accessed data
- Scale horizontally if needed

### 7.2 Rollback Plan

If issues arise:

**Trigger Conditions:**
- Model performance drops >5%
- Feature discrepancies >10%
- Production errors >1%
- Scoring time >30 seconds

**Rollback Steps:**
1. Switch scoring back to SQL pipeline
2. Investigate root cause
3. Fix issue in Python pipeline
4. Re-validate and re-deploy

**Rollback Time:** <1 hour (keep SQL scripts ready)

### 7.3 Monitoring

Post-deployment monitoring:

```python
# monitoring/feature_drift_monitor.py

def monitor_feature_drift():
    """
    Monitor for feature drift between training and production.
    Alert if distributions change significantly.
    """
    # Load training data statistics
    train_stats = pd.read_csv('Data/training_feature_stats.csv')

    # Get production data (last 7 days)
    prod_data = fetch_production_scores(days=7)
    prod_stats = calculate_statistics(prod_data)

    # Compare distributions
    for feature in train_stats['feature']:
        train_mean = train_stats.loc[train_stats['feature']==feature, 'mean'].values[0]
        prod_mean = prod_stats.loc[prod_stats['feature']==feature, 'mean'].values[0]

        drift_pct = abs(prod_mean - train_mean) / train_mean * 100

        if drift_pct > 20:  # 20% drift threshold
            send_alert(f"Feature drift detected: {feature} ({drift_pct:.1f}%)")
```

---

## 8. Risk Mitigation

### 8.1 Identified Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Feature mismatch (Python ≠ SQL) | Medium | High | Extensive validation, unit tests |
| Model performance degradation | Low | Critical | A/B testing, rollback plan |
| Performance issues (slow) | Medium | Medium | Profiling, optimization, caching |
| Memory errors (large data) | Low | Medium | Chunked processing, monitoring |
| Production bugs | Low | High | Thorough testing, shadow mode |
| Team adoption resistance | Medium | Low | Training, documentation |
| Timeline delays | Medium | Low | Phased approach, buffer time |

### 8.2 Mitigation Strategies

**Technical:**
- ✅ Extensive unit and integration tests
- ✅ Validation against SQL baseline
- ✅ Shadow mode deployment
- ✅ Monitoring and alerting
- ✅ Rollback capability

**Process:**
- ✅ Code reviews for all new code
- ✅ Documentation for team
- ✅ Training sessions
- ✅ Weekly progress reviews

**Organizational:**
- ✅ Executive sponsorship
- ✅ Cross-functional team (ML, DE, DevOps)
- ✅ Clear success criteria
- ✅ Stakeholder communication plan

---

## 9. Success Metrics

### 9.1 Technical Metrics

- ✅ **Feature Accuracy:** >95% match with SQL
- ✅ **Model Performance:** Gini within 1% of baseline
- ✅ **Processing Speed:** <10 minutes for training data
- ✅ **Scoring Latency:** <5 seconds per applicant
- ✅ **Memory Efficiency:** <16GB RAM for training
- ✅ **Test Coverage:** >80% code coverage
- ✅ **Bug Count:** <5 critical bugs in production

### 9.2 Business Metrics

- ✅ **Deployment Success:** Python pipeline in production
- ✅ **Downtime:** <1 hour during transition
- ✅ **Scoring Volume:** 1000+ applications/day supported
- ✅ **Team Velocity:** Feature experimentation time reduced by 50%
- ✅ **Maintenance Cost:** Reduced by 30% (less SQL maintenance)

### 9.3 Timeline Metrics

- ✅ **Phase 1-2:** Completed in 4 weeks
- ✅ **Phase 3-4:** Completed in 6 weeks
- ✅ **Phase 5-6:** Completed in 2 weeks
- ✅ **Total:** 12 weeks to production

---

## 10. Next Steps

### Immediate Actions (This Week)

1. ✅ Review this plan with stakeholders
2. ✅ Get approval to proceed
3. ✅ Set up development environment
4. ✅ Create project board (Jira/GitHub Projects)
5. ✅ Assign team roles

### Week 1 Actions

1. ✅ Create new SQL extraction script
2. ✅ Extract raw data (data_raw.csv)
3. ✅ Set up `QNBAnalytics_ML/feature_engineering.py`
4. ✅ Implement `BaseFeatureTransformer` class
5. ✅ Write initial tests

---

## Appendix

### A. Dependencies

Add to `requirements.txt`:

```
# Existing
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
category-encoders>=2.3.0
feature-engine>=1.4.0

# New (if needed)
pytest>=7.0.0  # For testing
memory_profiler>=0.60.0  # For profiling
```

### B. Glossary

- **WOE:** Weight of Evidence - encoding method for credit scoring
- **Gini:** Model discrimination metric (0-1, higher is better)
- **MKR:** Credit bureau report (Mərkəzi Kredit Reyestri)
- **FIN:** Tax identification number (anonymized)
- **CWPS:** Current Worst Payment Status
- **DRV:** Derived (suffix for transformed columns)
- **Layer 1/2/3:** Hierarchical ensemble layers

### C. References

- SQL Scripts: `scripts/1.sql` through `scripts/6.sql`
- Existing Pipeline: `QNBAnalytics_ML/` modules
- Training Notebook: `BOB_Scorecard_Training.ipynb`
- Root Cause Analysis: `docs/ROOT_CAUSE_ANALYSIS.md`

---

**End of Migration Plan**

For questions or clarifications, contact the ML Engineering team.
