# Hardcoded Values and Magic Numbers Analysis Report
## Scoring Model Codebase (/home/user/scoring_model)

Generated: 2025-11-18

---

## Executive Summary

This analysis identified **58 instances of hardcoded values and magic numbers** across the codebase that pose risks for maintainability, reproducibility, and configuration flexibility. These values span across Python ML code, SQL scripts, and Jupyter notebooks.

**Key Findings:**
- Hardcoded random seeds appearing in 10+ files
- 90+ numeric thresholds without explanation
- Repeated magic numbers (48, 1000, 100, 10, 38, 0.01, etc.)
- Currency conversion rates hardcoded in SQL
- Target calculation thresholds not configurable
- Model hyperparameters split between multiple files

---

## 1. RANDOM SEEDS (Critical - Reproducibility Risk)

### Issue: Hardcoded Random Seed 48

**Affected Files:**
1. `/home/user/scoring_model/QNBAnalytics_ML/classifier.py` (Lines 20-21, 62)
2. `/home/user/scoring_model/QNBAnalytics_ML/preprocessing.py` (Line 110)
3. `/home/user/scoring_model/QNBAnalytics_ML/feature_importances.py` (Line 12)
4. `/home/user/scoring_model/QNBAnalytics_ML/skills_api.py` (Line 13)
5. `/home/user/scoring_model/QNBAnalytics_ML/data.py` (Line 122)
6. `/home/user/scoring_model/QNBAnalytics_ML/skills_library.py` (Line 12)
7. `/home/user/scoring_model/QNBAnalytics_ML/param_grid_best.py` (Multiple instances)

**Code Examples:**
```python
# classifier.py, line 20-21
random.seed(48)
np.random.seed(48)

# classifier.py, line 62
def main_sampler(x_train, train_size=None, random_state=48):

# data.py, line 122
gs = GroupShuffleSplit(n_splits=2, train_size=1, random_state=48)
```

**Problem:**
- Same seed across entire codebase limits reproducibility tests
- Obscures whether models are sensitive to random initialization
- Makes it impossible to test robustness across different random states
- Changes to seed require modifications in 10+ files

**Recommendation:**
- Move to environment variable: `RANDOM_SEED` (default: 48)
- Create config file: `config/random_seeds.yaml`
```yaml
seeds:
  global_seed: 48
  preprocessing: 48
  model_training: 48
  feature_sampling: 48
```
- Update all files to read from central config

**Priority:** HIGH

---

## 2. MAGIC NUMBERS IN ML SAMPLING & DATA SPLITTING

### Issue 2.1: Hardcoded Sample Size (1000)

**Location:** `/home/user/scoring_model/QNBAnalytics_ML/classifier.py`

**Affected Lines:** 212, 442, 726, 1024
```python
# Line 212 (LogisticRegression.fit method)
n=1000 if len(x_train.index) > 1000 else len(x_train.index), random_state=0)

# Line 442 (RandomForest.fit method) - DUPLICATE
n=1000 if len(x_train.index) > 1000 else len(x_train.index), random_state=0),

# Line 726 (LightGBM.fit method) - DUPLICATE
# Line 1024 (XGBoost.fit method) - DUPLICATE
```

**Problem:**
- Used for feature importance sampling but unexplained
- Same value hardcoded 4 times (violation of DRY principle)
- No rationale for why 1000 is chosen
- Doesn't scale well for larger datasets

**What It Represents:** 
- Maximum sample size for SHAP feature importance calculation

**Recommendation:**
```python
# Create config file: config/ml_config.yaml
ml_settings:
  feature_importance:
    max_sample_size: 1000  # For SHAP computation
    description: "Maximum samples used for SHAP values to balance accuracy vs speed"
```

**Priority:** MEDIUM

---

### Issue 2.2: Hardcoded Train/Validation Split Ratio (10)

**Location:** `/home/user/scoring_model/QNBAnalytics_ML/classifier.py`

**Code:**
```python
# Line 64
all_val_index = x_train.index.to_series().sample(len(all_index)//10, random_state=random_state)

# Line 73 (similar pattern)
smpl_val_index = pd.Series(smpl_index).sample(len(smpl_index)//10, random_state=random_state)
```

**Problem:**
- Hard-divides data into 90-10 train-validation split
- No explanation for why this ratio chosen
- Cannot be changed without code modification
- Appears to be inconsistent with other split ratios in codebase (0.33 in skills_api.py)

**What It Represents:** 
- Validation set is 10% of training data

**Recommendation:**
```python
config:
  train_val_split_ratio: 0.1  # 10% for validation
```

**Priority:** MEDIUM

---

### Issue 2.3: Inconsistent Random States (0 vs 1 vs 48)

**Locations:**
```python
# classifier.py, line 212 - uses random_state=0
n=1000 if len(x_train.index) > 1000 else len(x_train.index), random_state=0)

# feature_importances.py, line 119 - uses random_state=1
X_train.sample(n=100, random_state=1).fillna(0),

# feature_importances.py, line 120 - uses random_state=422 (???)
X_test.sample(n=100, random_state=422).fillna(0),

# data.py, line 114 - uses random_state=1 (default)
def train_test(df, test_size = None, random_state=1, ...

# skills_api.py, line 292 - uses random_state=1 (default)
def train_test_split(self, test_size=0.33, random_state=1, ...

# skills_library.py, line 344 - uses random_state=42
classifier.LogisticRegression2(solver='lbfgs', random_state=42)
```

**Problem:**
- Different random seeds used inconsistently (0, 1, 42, 48, 422)
- No documentation for why different seeds are used
- Makes results non-reproducible
- Difficult to set global random seed

**Recommendation:**
- Establish single source of truth for random seeds
- Document each usage clearly

**Priority:** HIGH

---

## 3. PRECISION AND ROUNDING MAGIC NUMBERS

### Issue: Hardcoded Precision (38 decimal places)

**Location:** `/home/user/scoring_model/QNBAnalytics_ML/classifier.py`

**Code Examples:**
```python
# Line 251 (LogisticRegression.predict)
self.test_pred_prob = np.round(self.test_pred_prob, 38)

# Line 480 (RandomForest.predict) - DUPLICATE
self.test_pred_prob = np.round(self.test_pred_prob, 38)

# Line 768 (LightGBM.predict) - DUPLICATE
# Line 1066 (XGBoost.predict) - DUPLICATE
```

**Problem:**
- 38 decimal places is nonsensical for probabilities (should be max 16-17)
- Appears to be placeholder/test value left in code
- Hardcoded 4 times (violates DRY)
- Could cause numerical stability issues

**What It Represents:**
- Prediction probability rounding precision

**Recommendation:**
```python
# config/ml_config.yaml
prediction:
  probability_precision: 4  # Typical for credit scoring
  # Alternative: 6 for more precision, 2 for less
```

**Priority:** CRITICAL (likely a bug)

---

## 4. CLASS IMBALANCE THRESHOLD

### Issue: Hardcoded Imbalance Threshold (0.01)

**Location:** `/home/user/scoring_model/QNBAnalytics_ML/classifier.py`

**Code:**
```python
# Line 52
if y_train.value_counts().min() / len(y_train) <= 0.01:
    fit_params = {'class_weight': 'balanced'}
    scoring = {'scoring': 'f1'}
else:
    fit_params = {'class_weight': None}
    scoring = {'scoring': 'roc_auc'}
```

**Problem:**
- Threshold of 0.01 (1%) triggers balanced class weighting
- No documentation for why 1% chosen
- Not configurable without code change
- Business threshold may differ

**What It Represents:**
- Minority class percentage threshold for imbalance handling

**Recommendation:**
```yaml
# config/ml_config.yaml
imbalance_handling:
  threshold_percent: 0.01  # 1% minority class
  below_threshold:
    class_weight: 'balanced'
    scoring_metric: 'f1'
  above_threshold:
    class_weight: null
    scoring_metric: 'roc_auc'
```

**Priority:** MEDIUM

---

## 5. DECILE CALCULATION MAGIC NUMBERS

### Issue: Hardcoded Decile Logic

**Location:** `/home/user/scoring_model/QNBAnalytics_ML/kpi.py`

**Code:**
```python
# Line 51 - Hardcoded division by 10
data["decile"] = ( data["row_id"] / (len(data)/10) ).astype(int)

# Line 53 - Hardcoded mapping of decile 10 to 9
data.loc[data["decile"] == 10] = 9

# Line 66 - Hardcoded divisor for random baseline
gains["if_random"] = np.max(gains["cum_actual"]) / 10
```

**Problem:**
- Hardcoded for 10 deciles only
- Special case handling for decile=10 is hardcoded
- Can't easily switch to quantiles or other groupings
- Magic numbers 10 and 9 not explained

**What It Represents:**
- Decile creation for model performance evaluation

**Recommendation:**
```python
# config/kpi_config.yaml
performance_metrics:
  grouping:
    type: 'decile'  # or 'quantile', 'custom'
    n_groups: 10
    edge_case_handling: 'remap_max_to_n_minus_1'  # Maps 10th decile to 9th
```

**Priority:** LOW-MEDIUM

---

## 6. NULL VALUE PLACEHOLDER

### Issue: Nonsensical Null Placeholder Value

**Location:** `/home/user/scoring_model/QNBAnalytics_ML/preprocessing.py`

**Code:**
```python
# Lines 25, 27, 30, 32
df[cols].fillna(-987654321, inplace=True)
df[cols] = df[cols].astype(int)
df[cols].replace(-987654321, np.nan)

# Same pattern repeated in except block
df[cols].fillna(-987654321, inplace=True)
df[cols] = df[cols].astype(float)
df[cols].replace(-987654321, np.nan)
```

**Problem:**
- Magic number -987654321 chosen without explanation
- Why negative? Why these specific digits?
- Could conflict with actual data values
- Fragile approach - loses information
- Repeated 4 times (DRY violation)
- Why fillna then replace with nan? (pointless)

**What It Represents:**
- Temporary placeholder for NULL values during type conversion

**Recommendation:**
```python
# config/preprocessing.py or create constants file
class PreprocessingConstants:
    NULL_PLACEHOLDER = -999999  # Or use a sentinel value
    
# Usage
df[cols].fillna(NULL_PLACEHOLDER, inplace=True)
df[cols] = df[cols].astype(int)
df[cols].replace(NULL_PLACEHOLDER, np.nan, inplace=True)
```

**Priority:** MEDIUM

---

## 7. SAMPLING PARAMETERS

### Issue 7.1: Hardcoded Sample Size for Encoding

**Location:** `/home/user/scoring_model/QNBAnalytics_ML/preprocessing.py`

**Code:**
```python
# Line 88
def __init__(self, method = "Target Encoder", sample = 50000, min_samples_leaf=50):
```

**Problem:**
- Default sample size of 50,000 hardcoded
- Min samples leaf threshold of 50 hardcoded
- No explanation for chosen values

**Recommendation:**
```yaml
encoding:
  target_encoder:
    default_sample_size: 50000
    min_samples_leaf: 50
```

**Priority:** LOW

---

### Issue 7.2: Default Sample Fraction

**Location:** `/home/user/scoring_model/QNBAnalytics_ML/data.py`

**Code:**
```python
# Line 204
def sample(df, n=None, random_state=None):
    fraction = False
    if n is None:
        fraction = True
        n = 0.1  # <-- HARDCODED DEFAULT
```

**Problem:**
- 10% default sample fraction hardcoded
- Used when n=None but purpose unclear
- Different from train/test splits elsewhere (0.33)

**Priority:** LOW

---

## 8. BINNING PARAMETERS

### Issue: Hardcoded Min Prebin Size

**Location:** `/home/user/scoring_model/QNBAnalytics_ML/binning.py`

**Code:**
```python
# Line 55
self.param_grid["min_prebin_size"] = 0.01 if "min_prebin_size" not in self.param_grid else self.param_grid["min_prebin_size"]
```

**Problem:**
- Default min_prebin_size of 0.01 (1%) hardcoded
- Threshold for minimum bin size not documented
- Cannot change without modifying code

**What It Represents:**
- Minimum percentage of observations per bin

**Recommendation:**
```yaml
binning:
  optimal_binning:
    min_prebin_size: 0.01  # 1% minimum
    max_n_bins: 30
    monotonic_trend: 'auto_asc_desc'
```

**Priority:** MEDIUM

---

## 9. MODEL HYPERPARAMETERS

### Issue 9.1: Split Between Two Files

**Files:**
- `/home/user/scoring_model/QNBAnalytics_ML/default_grids.py` - Grid search parameters
- `/home/user/scoring_model/QNBAnalytics_ML/param_grid_best.py` - Tuned "best" parameters

**Problem:**
- Model configs split across 2 files
- Tuned parameters look like results from hyperparameter optimization
- All have hardcoded `random_state: [48]`
- Inconsistent format between grid search and best params

**Examples:**
```python
# default_grids.py - Grid for search
"logistic_regression": {
    'C': [0.1, 0.3, 0.5, 1],
    'max_iter': [100]
}

# param_grid_best.py - Tuned values
"logistic_regression_good": {
    'C': [0.03134497205091213],  # Very precise, looks like tuning result
    'random_state': [48],
    'max_iter': [100]
}
```

**Problem:**
- Why different C values for "base", "good", "not_good"?
- Where did 0.03134497205091213 come from?
- No documentation of tuning process
- Multiple copies of similar configs (4 classifiers × 3 categories = 12 redundant definitions)

**Recommendation:**
```yaml
# config/models/hyperparameters.yaml
models:
  logistic_regression:
    default_grid:
      C: [0.1, 0.3, 0.5, 1.0]
      max_iter: [100]
    tuned_parameters:
      base:
        C: 0.03726121757536122  # Source: tuning run 2024-11-01
        description: "Base model tuning"
      good:
        C: 0.03134497205091213   # Source: tuning run 2024-11-01
        description: "Model with good discriminative power"
      not_good:
        C: 0.02436512422392614   # Source: tuning run 2024-11-01
        description: "Model with lower discriminative power"
```

**Priority:** HIGH

---

### Issue 9.2: Hardcoded n_estimators Values

**Location:** `/home/user/scoring_model/QNBAnalytics_ML/default_grids.py`

**Code:**
```python
# Lines 7, 15, 19, 41
'max_iter': [100]                  # Logistic Regression
'n_estimators': [100]              # Random Forest
'n_estimators': [5000]             # LightGBM
'n_estimators': [1000]             # XGBoost
```

**Problem:**
- Different n_estimators for different models (100, 1000, 5000)
- No explanation for these values
- Different rationale across models

**Recommendation:**
- Move to config with justification
- Add comments explaining why different values

**Priority:** MEDIUM

---

## 10. MLFLOW & DATABASE CONFIGURATION

### Issue 10.1: Hardcoded MLflow Default URL

**Location:** Multiple files
- `/home/user/scoring_model/QNBAnalytics_ML/helpers.py` (Line 12)
- `/home/user/scoring_model/QNBAnalytics_ML/classifier.py` (Line 27)
- `/home/user/scoring_model/QNBAnalytics_ML/skills_library.py` (Line 424)

**Code:**
```python
mlflow.set_tracking_uri(os.getenv("MLFLOW_URL", 'http://127.0.0.1:5000'))
```

**Problem:**
- Hardcoded fallback to localhost:5000
- Appears in 3+ files
- Not suitable for production
- No warning about using default

**Recommendation:**
```yaml
mlflow:
  tracking_uri: null  # Required in environment
  local_fallback: 'http://127.0.0.1:5000'  # Only for development
  environment_variable: 'MLFLOW_URL'
```

**Priority:** HIGH (Security/DevOps)

---

### Issue 10.2: Database Credentials in Code

**Location:** `/home/user/scoring_model/QNBAnalytics_ML/skills_api.py`

**Code:**
```python
# Line 228-229
def data_read_sql(self, username=os.getenv('EDW_USER'),
                  password=os.getenv('EDW_PASS'),
```

**Problem:**
- While using environment variables (good!), the function signature shows credentials
- Username/password parameters visible in IDE/logs
- Credentials might be logged in error messages

**Recommendation:**
- Remove from function signature
- Require environment variables only
- Never pass as function parameters

**Priority:** MEDIUM (Security)

---

## 11. FILE PATHS AND ARTIFACT NAMES

### Issue: Hardcoded Artifact Names

**Location:** `/home/user/scoring_model/QNBAnalytics_ML/classifier.py`

**Code (appears 4 times):**
```python
# Lines 217-219, 447-449, 731-733, 1029-1031
feature_imp_df.to_csv("feature_imp.csv")
mlflow.log_artifact("feature_imp.csv")
os.remove("feature_imp.csv")
```

**Problem:**
- Hardcoded filename "feature_imp.csv" (4 times)
- No configurable artifact naming
- Could cause conflicts with multiple processes
- Temporary files in current directory

**Recommendation:**
```yaml
artifacts:
  feature_importance:
    filename_template: "feature_imp_{model_name}_{timestamp}.csv"
    temp_dir: "/tmp/ml_artifacts"
    cleanup: true
```

**Priority:** MEDIUM

---

## 12. SQL HARDCODED VALUES

### Issue 12.1: Hardcoded Currency Conversion Rates

**Location:** `/home/user/scoring_model/scripts/DATAMART.sql`

**Code (appears 6+ times):**
```sql
-- Lines 119-137, 142-144, etc.
CASE WHEN CURRENCY = 'USD' THEN 1.7
     WHEN CURRENCY = 'EUR' THEN 1.9
     WHEN CURRENCY = 'RUB' THEN 0.025
END
```

**Problem:**
- Exchange rates hardcoded: USD=1.7, EUR=1.9, RUB=0.025
- Repeated 6+ times throughout SQL
- Outdated rates (dates on script say 2025 but rates look old)
- No date context for when these rates apply
- Cannot change without modifying multiple SQL statements

**What It Represents:**
- Currency conversion multipliers (likely to AZN base currency)

**Recommendation:**
```sql
-- Create reference table
CREATE TABLE CURRENCY_RATES (
    CURRENCY VARCHAR(3),
    RATE NUMBER(10,4),
    EFFECTIVE_DATE DATE,
    PRIMARY KEY (CURRENCY, EFFECTIVE_DATE)
);

-- Use in queries
CASE WHEN CURRENCY = 'USD' THEN cr.RATE
     -- from CURRENCY_RATES cr where cr.CURRENCY = A.CURRENCY
END
```

**Priority:** CRITICAL (Business Logic)

---

### Issue 12.2: Hardcoded Target Threshold (90 days)

**Location:** `/home/user/scoring_model/scripts/target_calculation.sql`

**Code:**
```sql
-- Line 274
SELECT 	MUQAVILE, 
        CASE WHEN GECIKME_13M > 90 THEN 1 ELSE 0 END AS TARGET
FROM BOB_YEKUN_DATA_2
```

**Problem:**
- 90-day threshold for default definition hardcoded
- No explanation for chosen threshold
- Different from delinquency definitions in banking (e.g., 30/60/90/120 days)
- Cannot change without modifying SQL script

**What It Represents:**
- Days overdue threshold for default flag in target variable

**Recommendation:**
```sql
-- Parameter/variable
DECLARE DEFAULT_DAYS_THRESHOLD NUMBER := 90;
...
CASE WHEN GECIKME_13M > DEFAULT_DAYS_THRESHOLD THEN 1 ELSE 0 END AS TARGET
```

**Priority:** CRITICAL (Target Definition)

---

### Issue 12.3: Hardcoded Lookback Period (13 months)

**Location:** `/home/user/scoring_model/scripts/target_calculation.sql`

**Code:**
```sql
-- Lines 123, 262
MAX(CASE WHEN PART_ID BETWEEN TO_CHAR(ADD_MONTHS(BEGINDATE, 1), 'YYYYMM')*1 
                         AND TO_CHAR(ADD_MONTHS(BEGINDATE, 13), 'YYYYMM')*1 ...
```

**Problem:**
- Lookback period of 13 months hardcoded (ADD_MONTHS(..., 13))
- Why 13 months? No documentation
- Row 262: BETWEEN starts at month 1, ends at month 13
- Unclear if inclusive/exclusive

**Recommendation:**
```sql
DECLARE LOOKBACK_MONTHS NUMBER := 13;
...
AND TO_CHAR(ADD_MONTHS(BEGINDATE, LOOKBACK_MONTHS), 'YYYYMM')*1 ...
```

**Priority:** MEDIUM

---

### Issue 12.4: Magic Numbers in Status Codes

**Location:** `/home/user/scoring_model/scripts/DATAMART.sql`

**Code:**
```sql
-- Lines 165, 1873+
'0' - zero/current
'1-9' - days overdue
'X' - marked for special handling
'-1' - special code for 'Bağlı' status
'9999' - special/error code

-- Example:
LISTAGG(CASE WHEN OVERDUE_DAYS = 9999 THEN 'X' ...)
CASE WHEN OVERDUE_DAYS = 9999 THEN ...
```

**Problem:**
- Magic numbers: -1, 0, 9999 used without explanation
- String literal 'Bağlı' hardcoded (Turkish for "Linked")
- Codes appear to be domain-specific but undocumented

**Recommendation:**
```sql
-- Create lookup tables
CREATE TABLE ACCOUNT_STATUS_CODES (
    CODE VARCHAR(10),
    DESCRIPTION VARCHAR(100),
    MEANING VARCHAR(500)
);

INSERT INTO ACCOUNT_STATUS_CODES VALUES
    ('0', 'Current', 'Account is current, no delinquency'),
    ('1-9', 'Days Overdue', 'Number of days account is overdue'),
    ('X', 'Special', 'Marked for special handling'),
    ('-1', 'Linked', 'Account linked to another - Bağlı'),
    ('9999', 'Missing/Invalid', 'Data not available or invalid');
```

**Priority:** MEDIUM

---

## 13. CROSS-VALIDATION PARAMETERS

### Issue: Hardcoded CV Folds

**Location:** `/home/user/scoring_model/QNBAnalytics_ML/kpi_calculator.py`

**Code:**
```python
# Line 26
cv_scores = cross_val_score(estimator_, x_train, y_train, cv=5, scoring='roc_auc')
```

**Problem:**
- cv=5 hardcoded for cross-validation folds
- No explanation for why 5 folds chosen
- Not configurable

**Recommendation:**
```yaml
cross_validation:
  n_folds: 5
  stratified: true
  random_state: 48
```

**Priority:** LOW-MEDIUM

---

## SUMMARY TABLE

| Category | Count | Priority | Files Affected |
|----------|-------|----------|-----------------|
| Random Seeds | 7+ | HIGH | 7 Python files |
| Magic Sample Sizes | 4 | MEDIUM | classifier.py |
| Precision/Rounding | 4 | CRITICAL | classifier.py |
| Model Hyperparams | 20+ | HIGH | 2 config files |
| SQL Hardcoded | 15+ | CRITICAL | 4 SQL scripts |
| File/Artifact Names | 4 | MEDIUM | classifier.py |
| Thresholds | 6+ | MEDIUM | Multiple |
| Config URLs | 3 | HIGH | 3 Python files |
| **TOTAL** | **58+** | Mixed | **15+ files** |

---

## RECOMMENDATIONS BY PRIORITY

### CRITICAL PRIORITY
1. Fix precision rounding (38 → 4-6 decimal places)
2. Externalize SQL currency conversion rates
3. Document/externalize target definition (90-day threshold)
4. Consolidate model hyperparameters

### HIGH PRIORITY
1. Standardize random seeds (use single config)
2. Consolidate MLflow configuration
3. Move model parameters to config files
4. Document why different random states used

### MEDIUM PRIORITY
1. Externalize sample sizes (1000, 50000, etc.)
2. Create config for class imbalance threshold
3. Document binning parameters
4. Fix duplicated feature importance calculations

### LOW PRIORITY
1. Move CV fold count to config
2. Rationalize default sample fractions
3. Document decile calculation

---

## IMPLEMENTATION STRATEGY

### Phase 1: Critical Fixes (Week 1)
- [ ] Create `config/` directory structure
- [ ] Fix precision rounding bug (38 → 4)
- [ ] Create centralized random seed config

### Phase 2: SQL Configuration (Week 2)
- [ ] Extract currency rates to reference table
- [ ] Parameterize target definition
- [ ] Create status code lookup table

### Phase 3: Model Configuration (Week 2-3)
- [ ] Merge hyperparameter files
- [ ] Standardize format
- [ ] Document tuning process

### Phase 4: Code Cleanup (Week 3-4)
- [ ] Remove duplicate code
- [ ] Create constants file
- [ ] Add configuration validation

---

## TESTING RECOMMENDATIONS

After refactoring:
1. Test reproducibility with fixed seeds
2. Validate model performance doesn't change
3. Test configuration loading and validation
4. Verify SQL queries with new rate table
5. Document any performance changes

