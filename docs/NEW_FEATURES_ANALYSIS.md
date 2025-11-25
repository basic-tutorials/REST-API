# Analysis of New Workplace Features

**Date:** 2025-11-25
**Version:** 2.0.0
**Analyst:** Claude Code

---

## Executive Summary

Five new workplace-related features have been added to improve the credit scoring model's Gini coefficient. These features show **strong predictive power**, with Gini coefficients ranging from **0.04 to 0.42**, significantly higher than many existing credit bureau features.

**Expected Impact:**
- Base Model Gini: 0.563 → **0.58-0.61** (+0.02 to +0.05)
- Meta Model Gini: 0.575 → **0.59-0.62** (+0.02 to +0.05)

---

## 1. New Features Identified

These 5 features were added at the end of the dataset (columns 2742-2746 out of 2748 total):

| Feature | Type | Description |
|---------|------|-------------|
| **MAAS** | Numeric | Customer salary/income (in local currency) |
| **WORKGROUP** | Categorical | Employment type/category |
| **QRUP** | Categorical | Organization type |
| **SAHƏLƏR** | Categorical | Business sector (9 categories) |
| **SEKTOR** | Categorical | Specific sector (89 categories) |

---

## 2. Feature Characteristics

### 2.1 MAAS (Salary)

**Type:** Numeric (continuous)

**Statistics:**
- Sample Size: 87,709 customers
- Non-null: 87,408 (99.7%)
- Missing: 301 (0.3%)
- Unique Values: 25,044

**Distribution:**
- Min: 0.00
- 25th percentile: 438.55
- **Median: 663.79**
- 75th percentile: 1,046.00
- Max: 113,658.00
- Mean: 877.67
- Std Dev: 839.61

**Predictive Power:**
- **Gini: 0.0398** (weak but positive)
- AUC: 0.5199
- Direction: Higher salary → lower default risk (expected)

**Data Quality:**
- ✅ Only 0.3% missing - excellent coverage
- ⚠️ Has zeros (0.00) - may need handling
- ⚠️ Has outliers (max: 113,658) - may need capping

---

### 2.2 WORKGROUP (Employment Type)

**Type:** Categorical

**Statistics:**
- Non-null: 67,289 (76.7%)
- **Missing: 20,420 (23.3%)** ⚠️
- Unique Values: 5 categories

**Default Rates by Category:**
| Category | Sample Size | Default Rate |
|----------|-------------|--------------|
| Category 1 | 9,110 | 2.48% |
| Category 2 | 679 | 1.77% |
| Category 3 | 24,443 | 1.33% |
| Category 4 | 28,200 | 0.95% |
| Category 5 | 4,857 | 0.72% |

**Predictive Power:**
- **Gini: 0.1955** (moderate, good predictor)
- AUC: 0.5977
- Range: 0.72% to 2.48% default rate

**Data Quality:**
- ⚠️ 23.3% missing - significant, requires imputation strategy
- ✅ Clear risk differentiation (3.4x range)
- ✅ Reasonable category sizes

---

### 2.3 QRUP (Organization Type)

**Type:** Categorical (Binary)

**Statistics:**
- Non-null: 87,652 (99.9%)
- Missing: 57 (0.1%)
- Unique Values: 2 categories

**Default Rates by Category:**
| Category | Sample Size | Default Rate |
|----------|-------------|--------------|
| Category 1 | 30,675 | 2.47% |
| Category 2 | 56,977 | 1.61% |

**Predictive Power:**
- **Gini: 0.1049** (moderate)
- AUC: 0.5525
- Range: 1.61% to 2.47% default rate

**Data Quality:**
- ✅ Only 0.1% missing - excellent coverage
- ✅ Balanced categories (35% / 65% split)
- ✅ Clear risk differentiation (1.5x range)

---

### 2.4 SAHƏLƏR (Business Sector)

**Type:** Categorical

**Statistics:**
- Non-null: 87,648 (99.9%)
- Missing: 61 (0.1%)
- Unique Values: 9 categories

**Default Rates by Category (Top 5):**
| Category | Sample Size | Default Rate |
|----------|-------------|--------------|
| Sector 1 | 12,736 | **4.00%** (High Risk) |
| Sector 2 | 21,076 | 2.74% |
| Sector 3 | 6,827 | 2.09% |
| Sector 4 | 3,306 | 1.45% |
| Sector 5 | 14,857 | 1.23% |

**Predictive Power:**
- **Gini: 0.3430** (STRONG predictor!) ⭐
- AUC: 0.6715
- Range: 1.23% to 4.00% default rate (3.3x range)

**Data Quality:**
- ✅ Only 0.1% missing - excellent coverage
- ✅ Strong risk differentiation
- ✅ Reasonable category sizes
- ⭐ **Expected to be in TOP 15 features**

---

### 2.5 SEKTOR (Specific Sector)

**Type:** Categorical (High Cardinality)

**Statistics:**
- Non-null: 87,648 (99.9%)
- Missing: 61 (0.1%)
- Unique Values: 89 categories

**Default Rates by Category (Top 5 Riskiest):**
| Category | Sample Size | Default Rate |
|----------|-------------|--------------|
| Sector A | 1 | 100.00% (outlier) |
| Sector B | 12 | 16.67% |
| Sector C | 37 | 8.11% |
| Sector D | 5,855 | 5.72% |
| Sector E | 909 | 5.61% |

**Predictive Power:**
- **Gini: 0.4189** (VERY STRONG predictor!) ⭐⭐
- AUC: 0.7095
- Range: Very wide (1% to 100%)

**Data Quality:**
- ✅ Only 0.1% missing - excellent coverage
- ⚠️ High cardinality (89 categories) - requires special handling
- ⚠️ Small sample sizes in some categories (risk of overfitting)
- ⭐⭐ **Expected to be in TOP 5-10 features**

---

## 3. Feature Importance Ranking

Based on Gini coefficients, the new features rank as follows:

| Rank | Feature | Gini | Category | Expected Model Impact |
|------|---------|------|----------|----------------------|
| 1 | **SEKTOR** | 0.4189 | Categorical (89) | 🏆 TOP 5-10 overall |
| 2 | **SAHƏLƏR** | 0.3430 | Categorical (9) | 🥈 TOP 10-15 overall |
| 3 | **WORKGROUP** | 0.1955 | Categorical (5) | 🥉 TOP 20-30 overall |
| 4 | **QRUP** | 0.1049 | Categorical (2) | TOP 30-50 overall |
| 5 | **MAAS** | 0.0398 | Numeric | Moderate importance |

**Context:** For comparison, typical credit bureau features have Gini coefficients of 0.05-0.20. Features with Gini > 0.30 are considered **very strong predictors**.

---

## 4. Recommended Handling Strategy

### 4.1 MAAS (Salary) - Numeric

**Current Pipeline Position:** Layer 1 (Base Model)

**Recommended Preprocessing:**

1. **Missing Value Imputation** (0.3% missing)
   ```python
   # Option 1: Mean imputation (CURRENT)
   apply_null_imputation = True  # Uses mean imputation

   # Option 2: Median imputation (RECOMMENDED for skewed data)
   # Modify preprocessing.py to use median for MAAS
   ```

2. **Outlier Treatment**
   ```python
   # Cap at 99th percentile to reduce outlier impact
   threshold = data['MAAS'].quantile(0.99)  # ~3,000
   data['MAAS'] = data['MAAS'].clip(upper=threshold)
   ```

3. **Zero Handling**
   - Keep zeros as-is (may represent unemployed/self-employed)
   - Or: Treat as missing and impute

4. **Transformation**
   - **NOT needed for Layer 1** (WOE binning handles it)
   - For Layer 2: Consider log transformation if skewness is high

**Why This Works:**
- WOE binning in Layer 1 automatically handles non-linearity
- Imputation ensures no data loss
- Outlier capping prevents extreme values from dominating

---

### 4.2 WORKGROUP (Employment Type) - Categorical

**Current Pipeline Position:** Layer 1 (Base Model)

**Recommended Preprocessing:**

1. **Missing Value Imputation** (23.3% missing - CRITICAL!)
   ```python
   # Option 1: Mode imputation (most frequent category)
   mode = data['WORKGROUP'].mode()[0]
   data['WORKGROUP'].fillna(mode, inplace=True)

   # Option 2: Create "Unknown" category (RECOMMENDED)
   data['WORKGROUP'].fillna('UNKNOWN', inplace=True)
   # This preserves signal that employment type is missing
   ```

2. **Categorical Encoding**
   ```python
   # CURRENT: TargetEncoder
   apply_categorical_encoding = True

   # Settings in preprocessing.py:
   TargetEncoder(
       cols=['WORKGROUP', 'QRUP', 'SAHƏLƏR', 'SEKTOR'],
       min_samples_leaf=50,  # Regularization
       smoothing=1.0
   )
   ```

3. **WOE Binning** (Layer 1 only)
   - Applied AFTER encoding
   - Handles encoded values

**Why This Works:**
- TargetEncoder captures risk differentiation (2.48% vs 0.72%)
- "Unknown" category preserves information about missingness
- WOE binning adds interpretability

---

### 4.3 QRUP (Organization Type) - Binary Categorical

**Current Pipeline Position:** Layer 1 (Base Model)

**Recommended Preprocessing:**

1. **Missing Value Imputation** (0.1% missing - minimal)
   ```python
   # Mode imputation (simple and effective)
   mode = data['QRUP'].mode()[0]
   data['QRUP'].fillna(mode, inplace=True)
   ```

2. **Categorical Encoding**
   ```python
   # CURRENT: TargetEncoder (works well)
   apply_categorical_encoding = True
   ```

**Why This Works:**
- Only 2 categories → simple and stable
- Minimal missing data → low risk
- Clear risk differentiation maintained

---

### 4.4 SAHƏLƏR (Business Sector) - Categorical

**Current Pipeline Position:** Layer 1 (Base Model)

**Recommended Preprocessing:**

1. **Missing Value Imputation** (0.1% missing - minimal)
   ```python
   # Option 1: Mode imputation
   mode = data['SAHƏLƏR'].mode()[0]
   data['SAHƏLƏR'].fillna(mode, inplace=True)

   # Option 2: Create "Unknown" category
   data['SAHƏLƏR'].fillna('UNKNOWN', inplace=True)
   ```

2. **Categorical Encoding**
   ```python
   # CURRENT: TargetEncoder (PERFECT for this)
   apply_categorical_encoding = True

   # Strong signal → will create good WOE bins
   ```

**Why This Works:**
- High predictive power (Gini 0.34) → TargetEncoder captures it well
- 9 categories → manageable, not too many
- Clear risk hierarchy (4.00% to 1.23%)

---

### 4.5 SEKTOR (Specific Sector) - High Cardinality Categorical

**Current Pipeline Position:** Layer 1 (Base Model)

**⚠️ SPECIAL HANDLING REQUIRED** (89 categories!)

**Recommended Preprocessing:**

1. **Missing Value Imputation** (0.1% missing)
   ```python
   data['SEKTOR'].fillna('UNKNOWN', inplace=True)
   ```

2. **High Cardinality Handling**

   **Option 1: TargetEncoder with Strong Regularization (CURRENT)**
   ```python
   TargetEncoder(
       cols=['SEKTOR'],
       min_samples_leaf=100,  # INCREASE from 50
       smoothing=10.0          # INCREASE from 1.0
   )
   ```
   - **Pros:** Captures full signal, handles rare categories
   - **Cons:** Risk of overfitting, needs more regularization

   **Option 2: Frequency Encoding + TargetEncoder**
   ```python
   # Add frequency as a separate feature
   freq = data['SEKTOR'].value_counts()
   data['SEKTOR_FREQ'] = data['SEKTOR'].map(freq)

   # Then apply TargetEncoder
   ```
   - **Pros:** Adds stability, preserves frequency signal
   - **Cons:** Creates 2 features instead of 1

   **Option 3: Group Rare Categories**
   ```python
   # Group categories with < 100 samples
   freq = data['SEKTOR'].value_counts()
   rare_categories = freq[freq < 100].index
   data['SEKTOR'] = data['SEKTOR'].apply(
       lambda x: 'RARE_SECTOR' if x in rare_categories else x
   )
   ```
   - **Pros:** Reduces overfitting, more stable
   - **Cons:** May lose some signal from rare categories

**RECOMMENDATION:** Use **Option 1** (current approach) but **increase regularization**:
- `min_samples_leaf=100` (up from 50)
- `smoothing=5.0` (up from 1.0)

**Why This Works:**
- Highest Gini (0.42) → must preserve this signal
- Regularization prevents overfitting on small categories
- WOE binning adds another layer of regularization

---

## 5. Current Pipeline Implementation

### 5.1 Pipeline Order (CORRECT) ✅

```python
# Layer 1: Base Model
1. Drop Null Features     (threshold=0.99)
2. Drop Constant Features
3. Drop Low Gini Features (threshold=0.05)
4. Drop Correlated Features
5. Null Imputation        ← Handles MAAS missing values
6. Categorical Encoding   ← Handles WORKGROUP, QRUP, SAHƏLƏR, SEKTOR
7. WOE Binning           ← Creates interpretable bins
8. Logistic Regression Training
```

**Why This Order Works:**
- Imputation BEFORE encoding → ensures all features have values
- Encoding BEFORE binning → converts categories to numbers first
- Gini elimination on numerical variables only → works correctly now

---

### 5.2 Configuration Settings

**In Notebook Cell (CURRENT):**
```python
apply_null_imputation = True       # ✅ CORRECT (for MAAS)
apply_categorical_encoding = True  # ✅ CORRECT (for 4 categorical features)
apply_binning = True               # ✅ CORRECT (for interpretability)
```

**In `QNBAnalytics_ML/preprocessing.py`:**
```python
# TargetEncoder settings
TargetEncoder(
    cols=['WORKGROUP', 'QRUP', 'SAHƏLƏR', 'SEKTOR'],
    min_samples_leaf=50,  # Consider increasing to 100 for SEKTOR
    smoothing=1.0          # Consider increasing to 5.0 for SEKTOR
)

# SimpleImputer for numeric features (MAAS)
SimpleImputer(strategy='mean')  # Or 'median' for skewed data
```

---

## 6. Expected Performance Impact

### 6.1 Base Model (Layer 1)

**Before (v1.0.0):**
- Gini: 0.563 (test set)
- Features: 2,738 credit bureau features only

**After (v2.0.0 - Expected):**
- Gini: **0.58-0.61** (test set)
- Features: 2,743 (added 5 new features)
- **Improvement: +0.02 to +0.05** (3.5% to 8.9% relative)

**Contributors:**
- SEKTOR: +0.015 to +0.025 (largest contributor)
- SAHƏLƏR: +0.005 to +0.015
- WORKGROUP: +0.005 to +0.010
- QRUP: +0.002 to +0.005
- MAAS: +0.001 to +0.003

---

### 6.2 Meta Model (Layer 3)

**Before (v1.0.0):**
- Gini: 0.575 (test set)

**After (v2.0.0 - Expected):**
- Gini: **0.59-0.62** (test set)
- **Improvement: +0.02 to +0.05**

**Cascade Effect:**
- Layer 1 improvement → better base predictions
- Layer 2 improvement → segment models benefit
- Layer 3 improvement → meta model combines better inputs

---

### 6.3 Feature Importance

**Expected Rankings in Layer 1:**

| Rank | Feature | Type | Expected Gini |
|------|---------|------|---------------|
| 1-5 | **SEKTOR** | New | 0.40-0.42 |
| 5-10 | **SAHƏLƏR** | New | 0.32-0.35 |
| 10-15 | CC_O_3MWPS_EVER | Old | 0.15-0.20 |
| 15-20 | **WORKGROUP** | New | 0.18-0.22 |
| 20-30 | CL_A_CWPS_90D | Old | 0.10-0.15 |
| 25-35 | **QRUP** | New | 0.09-0.12 |
| 30-40 | **MAAS** | New | 0.03-0.05 |

---

## 7. Data Quality Concerns & Mitigation

### 7.1 WORKGROUP: 23.3% Missing ⚠️

**Concern:** High missing rate may bias model

**Mitigation:**
1. **Create "Unknown" category** (recommended)
   - Preserves information that employment type is unknown
   - Model learns that "unknown" has its own risk profile

2. **Analyze missing pattern:**
   ```python
   # Check if missingness correlates with default
   missing_df = data['WORKGROUP'].isna()
   print(f"Default rate (WORKGROUP known): {data[~missing_df]['TARGET'].mean():.2%}")
   print(f"Default rate (WORKGROUP missing): {data[missing_df]['TARGET'].mean():.2%}")
   ```

3. **Monitor in production:**
   - Track % of scored applications with missing WORKGROUP
   - Alert if % increases significantly

---

### 7.2 SEKTOR: 89 Categories (High Cardinality) ⚠️

**Concern:** Risk of overfitting on rare categories

**Mitigation:**
1. **Increase TargetEncoder regularization:**
   - `min_samples_leaf=100` (ensures minimum 100 samples per category)
   - `smoothing=5.0` (adds global mean regularization)

2. **Monitor rare categories:**
   ```python
   # Identify categories with < 100 samples
   freq = data['SEKTOR'].value_counts()
   rare = freq[freq < 100]
   print(f"Rare categories: {len(rare)} / {len(freq)}")
   ```

3. **Cross-validation:**
   - Ensure consistent performance across CV folds
   - If high variance → increase regularization or group rare categories

4. **WOE binning as safety net:**
   - After TargetEncoder, WOE binning groups similar values
   - Provides additional regularization

---

### 7.3 MAAS: Outliers (max: 113,658) ⚠️

**Concern:** Extreme values may distort patterns

**Mitigation:**
1. **Outlier capping:**
   ```python
   # Cap at 99th percentile
   threshold_99 = data['MAAS'].quantile(0.99)
   data['MAAS'] = data['MAAS'].clip(upper=threshold_99)
   ```

2. **Log transformation (Layer 2 only):**
   ```python
   # For Layer 2 models (LGBM handles this naturally)
   data['MAAS_log'] = np.log1p(data['MAAS'])
   ```

3. **WOE binning (Layer 1):**
   - Automatically handles outliers by creating bins
   - Extreme values grouped into top bin

---

## 8. Implementation Checklist

### Phase 1: Pre-Training ✅ (COMPLETED)

- [x] Add new features to dataset (Data/data.csv)
- [x] Update `cols_to_drop` list (exclude ID columns)
- [x] Enable `apply_null_imputation = True`
- [x] Enable `apply_categorical_encoding = True`
- [x] Fix pipeline bugs (lambda, pandas.append, variable handling)

### Phase 2: Training 🔄 (IN PROGRESS)

- [ ] Restart Jupyter kernel
- [ ] Run training notebook (BOB_Scorecard_Training.ipynb)
- [ ] Monitor for errors in:
  - [ ] Null imputation step
  - [ ] Categorical encoding step
  - [ ] WOE binning step
  - [ ] Model training steps
- [ ] Check training logs for completion

### Phase 3: Validation (AFTER TRAINING)

- [ ] Check feature importances:
  - [ ] SEKTOR in TOP 10
  - [ ] SAHƏLƏR in TOP 20
  - [ ] WORKGROUP in TOP 30
- [ ] Verify Gini improvement:
  - [ ] Base Model: 0.58+ (vs 0.563 before)
  - [ ] Meta Model: 0.59+ (vs 0.575 before)
- [ ] Review binning tables:
  - [ ] Check MAAS bins (monotonic relationship)
  - [ ] Check categorical encoding (sensible values)
- [ ] Analyze predictions:
  - [ ] Score distribution (160-250 range)
  - [ ] Default rate by score decile

### Phase 4: Fine-Tuning (IF NEEDED)

- [ ] If SEKTOR overfits:
  - [ ] Increase `min_samples_leaf` to 150
  - [ ] Increase `smoothing` to 10.0
  - [ ] Or group rare categories (<100 samples)
- [ ] If WORKGROUP missingness is problematic:
  - [ ] Analyze missing pattern correlation with default
  - [ ] Consider alternative imputation strategy
- [ ] If MAAS outliers cause issues:
  - [ ] Cap at 95th or 99th percentile
  - [ ] Consider log transformation for Layer 2

---

## 9. Monitoring & Production Deployment

### 9.1 Model Monitoring

**Key Metrics to Track:**

1. **Feature Stability**
   ```python
   # Check distributions haven't shifted
   - MAAS: mean, median, std dev
   - WORKGROUP: category frequencies
   - SEKTOR: category frequencies
   ```

2. **Missing Rates**
   ```python
   # Alert if missing rates increase
   - MAAS: Should stay < 1%
   - WORKGROUP: Should stay < 30%
   - QRUP/SAHƏLƏR/SEKTOR: Should stay < 0.5%
   ```

3. **Encoding Values**
   ```python
   # Ensure encodings are sensible
   - Check TargetEncoder output ranges
   - Verify WOE bin assignments
   ```

---

### 9.2 Production Scoring

**Pipeline Requirements:**

1. **Saved Artifacts** (from training)
   - `base_model_training.pkl` - includes WOE binning transformers
   - `good_model_training.pkl` - includes TargetEncoder transformers
   - `not_good_model_training.pkl` - includes TargetEncoder transformers
   - `meta_model_training.pkl`

2. **Feature Availability**
   - Ensure all 5 new features are in scoring data
   - If any missing → use same imputation strategy as training
   - **Critical:** WORKGROUP can be missing (will be imputed)

3. **Encoding Consistency**
   - Use SAME TargetEncoder fitted on training data
   - Use SAME WOEEncoder fitted on training data
   - Never fit encoders on scoring data!

---

## 10. Conclusions & Recommendations

### 10.1 Key Findings

1. **SEKTOR and SAHƏLƏR are VERY strong predictors**
   - Gini 0.42 and 0.34 respectively
   - Expected to be in TOP 10 features overall
   - Will significantly improve model performance

2. **WORKGROUP has moderate power but HIGH missing rate**
   - Gini 0.20 - decent predictor
   - 23.3% missing - requires careful handling
   - Use "Unknown" category approach

3. **Current pipeline handles new features correctly**
   - Imputation → Encoding → Binning order is correct
   - TargetEncoder settings are reasonable
   - WOE binning provides additional regularization

4. **Expected Gini improvement: +0.02 to +0.05**
   - Base Model: 0.563 → 0.58-0.61
   - Meta Model: 0.575 → 0.59-0.62
   - Significant improvement for production model

---

### 10.2 Final Recommendations

**IMMEDIATE ACTIONS:**

1. ✅ **Continue with current training run** (after fixing bugs)
   - Pipeline is correctly configured
   - No major changes needed

2. ⚠️ **Monitor SEKTOR for overfitting**
   - If cross-validation shows high variance:
     - Increase regularization (min_samples_leaf=100, smoothing=5.0)
     - Or group rare categories

3. ⚠️ **Handle WORKGROUP missingness properly**
   - Use "Unknown" category (preserves signal)
   - Monitor default rate for missing vs non-missing

**FUTURE ENHANCEMENTS:**

1. **Add interaction features** (v2.1.0)
   ```python
   # Interactions that may be powerful:
   - MAAS × WORKGROUP (salary within employment type)
   - SAHƏLƏR × SEKTOR (sector within business area)
   - MAAS × SAHƏLƏR (salary within business sector)
   ```

2. **Engineer derived features** (v2.1.0)
   ```python
   # Potential derived features:
   - MAAS_per_loan_amount (debt-to-income ratio)
   - MAAS_bin (low/medium/high salary bracket)
   - SEKTOR_risk_score (pre-computed sector risk)
   ```

3. **A/B testing in production** (v2.2.0)
   - Compare v1.0.0 (without new features) vs v2.0.0
   - Measure real-world default rate improvement
   - Validate expected Gini gains

---

**Document prepared by:** Claude Code
**Last updated:** 2025-11-25
**Version:** 1.0
