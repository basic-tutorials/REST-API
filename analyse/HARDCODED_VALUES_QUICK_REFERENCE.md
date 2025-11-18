# Quick Reference: Hardcoded Values Found

## Top 10 Issues

### 1. PRECISION BUG - CRITICAL
**File:** `QNBAnalytics_ML/classifier.py` (Lines 251, 480, 768, 1066)
```python
np.round(self.test_pred_prob, 38)  # WRONG: 38 decimal places!
```
**Should be:** 4-6 decimal places  
**Impact:** Could cause numerical instability  
**Fix Priority:** Immediate

---

### 2. RANDOM SEED 48 - HIGH
**Files:** 7+ files across codebase  
**Hardcoded Locations:**
- `classifier.py:20-21` - `random.seed(48)`, `np.random.seed(48)`
- `preprocessing.py:110` - `random.seed(48)`
- `feature_importances.py:12` - `np.random.seed(48)`
- `skills_api.py:13` - `np.random.seed(48)`
- `data.py:122` - `random_state=48`
- `skills_library.py:12` - `np.random.seed(48)`
- `param_grid_best.py:*` - All models: `'random_state': [48]`

**Issue:** Same seed everywhere, hard to test robustness  
**Solution:** Use environment variable `RANDOM_SEED`

---

### 3. FEATURE IMPORTANCE SAMPLE SIZE - MEDIUM
**File:** `QNBAnalytics_ML/classifier.py` (Lines 212, 442, 726, 1024)
```python
n=1000 if len(x_train.index) > 1000 else len(x_train.index)
```
**Issue:** Hardcoded 4 times, violates DRY  
**Value:** 1000 samples for SHAP calculation

---

### 4. INCONSISTENT RANDOM STATES - HIGH
**Found:** 0, 1, 42, 48, 422 in different places  
**Files:**
- `classifier.py:212` → `random_state=0`
- `feature_importances.py:119` → `random_state=1`
- `feature_importances.py:120` → `random_state=422` (suspicious)
- `skills_library.py:344` → `random_state=42`
- `kpi_calculator.py:26` → `cv=5` (hardcoded CV folds)

---

### 5. CURRENCY CONVERSION RATES - CRITICAL
**File:** `scripts/DATAMART.sql` (Lines 119-137, 142-144, etc.)
```sql
CASE WHEN CURRENCY = 'USD' THEN 1.7
     WHEN CURRENCY = 'EUR' THEN 1.9
     WHEN CURRENCY = 'RUB' THEN 0.025
```
**Issue:** Hardcoded 6+ times, rates outdated?  
**Solution:** Create `CURRENCY_RATES` reference table

---

### 6. TARGET THRESHOLD - CRITICAL
**File:** `scripts/target_calculation.sql` (Line 274)
```sql
CASE WHEN GECIKME_13M > 90 THEN 1 ELSE 0 END AS TARGET
```
**Issue:** 90-day default definition hardcoded  
**Impact:** Controls target variable definition  
**Solution:** Parameterize as `DEFAULT_DAYS_THRESHOLD`

---

### 7. LOOKBACK PERIOD - MEDIUM
**File:** `scripts/target_calculation.sql` (Lines 123, 262)
```sql
ADD_MONTHS(BEGINDATE, 13)  -- Hardcoded 13 months
```
**Issue:** Why 13? No documentation  
**Solution:** Use `DECLARE LOOKBACK_MONTHS NUMBER := 13;`

---

### 8. CLASS IMBALANCE THRESHOLD - MEDIUM
**File:** `classifier.py` (Line 52)
```python
if y_train.value_counts().min() / len(y_train) <= 0.01:
    # Use balanced class weights
```
**Issue:** 1% threshold hardcoded, not configurable

---

### 9. NULL PLACEHOLDER VALUE - MEDIUM
**File:** `preprocessing.py` (Lines 25, 27, 30, 32)
```python
df[cols].fillna(-987654321, inplace=True)
```
**Issue:** Nonsensical value, repeated 4 times, DRY violation

---

### 10. ARTIFACT NAMING - MEDIUM
**File:** `classifier.py` (Lines 217-219, 447-449, 731-733, 1029-1031)
```python
feature_imp_df.to_csv("feature_imp.csv")  # Hardcoded 4 times
```
**Issue:** No versioning, process conflicts possible

---

## By File Summary

### `QNBAnalytics_ML/classifier.py` (30+ issues)
- Lines 20-21: `random.seed(48)` + `np.random.seed(48)`
- Line 27: `mlflow.set_tracking_uri(os.getenv("MLFLOW_URL", 'http://127.0.0.1:5000'))`
- Line 52: `<= 0.01` (class imbalance threshold)
- Line 62: `def main_sampler(..., random_state=48)`
- Line 64: `//10` (train/val split ratio)
- Line 73: `//10` (duplicate)
- Lines 212, 442, 726, 1024: `n=1000` (feature importance sample)
- Lines 217-219, 447-449, 731-733, 1029-1031: `"feature_imp.csv"` (artifact names)
- Lines 251, 480, 768, 1066: `np.round(..., 38)` (PRECISION BUG!)
- Line 379: `max_depth=10, n_estimators=200` (RF hyperparams)
- Line 906: `.get('early_stopping_rounds', 10)` (default rounds)

### `QNBAnalytics_ML/preprocessing.py`
- Lines 25, 27, 30, 32: `-987654321` (NULL placeholder, 4x)
- Line 88: `sample = 50000, min_samples_leaf=50`
- Line 110: `random.seed(48)`

### `QNBAnalytics_ML/data.py`
- Line 114: `random_state=1` (default)
- Line 122: `random_state=48`
- Line 204: `n = 0.1` (default sample fraction)

### `QNBAnalytics_ML/feature_importances.py`
- Line 12: `np.random.seed(48)`
- Line 119: `random_state=1`
- Line 120: `random_state=422` (???)
- Lines 150, 157: `max_evals = 2001` (SHAP iterations)

### `QNBAnalytics_ML/skills_api.py`
- Line 13: `np.random.seed(48)`
- Line 228-229: `username=os.getenv('EDW_USER'), password=os.getenv('EDW_PASS')`
- Line 292: `test_size=0.33`
- Line 565: `thresh=100000, nprocs=5, max_n_bins=30, min_bin_size=0.0001`

### `QNBAnalytics_ML/kpi.py`
- Line 51: `/ (len(data)/10)` (hardcoded 10 deciles)
- Line 53: `data.loc[data["decile"] == 10]=9` (edge case)
- Line 66: `/ 10` (hardcoded divisor)

### `QNBAnalytics_ML/kpi_calculator.py`
- Line 26: `cv=5` (CV folds hardcoded)

### `QNBAnalytics_ML/default_grids.py`
- Line 6: `'C': [0.1, 0.3, 0.5, 1]`
- Line 7: `'max_iter': [100]`
- Line 15: `'n_estimators': [100]` (RF)
- Line 19: `'n_estimators': [5000]` (LGB)
- Line 41: `'n_estimators': [1000]` (XGB)

### `QNBAnalytics_ML/param_grid_best.py`
- Multiple lines with tuned hyperparameters
- All models have `'random_state': [48]`
- Values look like optimization results (very precise floats)

### `QNBAnalytics_ML/binning.py`
- Line 55: `0.01` (min_prebin_size default)

### `scripts/DATAMART.sql`
- Lines 119-137, 142-144: Currency conversion rates (USD=1.7, EUR=1.9, RUB=0.025)
- Lines 165, 1873+: Status codes (-1, 0, 9999, 'X', 'Bağlı')

### `scripts/target_calculation.sql`
- Line 274: `WHEN GECIKME_13M > 90 THEN 1` (90-day threshold)
- Lines 123, 262: `ADD_MONTHS(..., 13)` (lookback period)

---

## Migration Checklist

- [ ] Fix precision rounding (38 → 4 or 6)
- [ ] Create config file with:
  - [ ] Random seeds
  - [ ] Sample sizes
  - [ ] Thresholds
  - [ ] Model hyperparameters
- [ ] Create SQL reference tables:
  - [ ] CURRENCY_RATES
  - [ ] ACCOUNT_STATUS_CODES
- [ ] Parameterize SQL scripts:
  - [ ] Target definition
  - [ ] Lookback period
- [ ] Consolidate hyperparameter files
- [ ] Remove duplicate code (SHAP sampling, etc.)
- [ ] Add documentation for all magic numbers

