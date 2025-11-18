# CODE DUPLICATION - QUICK REFERENCE GUIDE

## Files with Duplication Issues

### CRITICAL - 4 Model Classes (60-75% identical)
**File:** `/home/user/scoring_model/QNBAnalytics_ML/classifier.py`

**Classes:**
- `LogisticRegression` (Lines 80-310)
- `RandomForest` (Lines 313-539)
- `LGBM` (Lines 542-826)
- `XGBoost` (Lines 829-1126)

**Identical Methods Across All 4 Classes:**
1. `__init__()` - 95% identical
2. `select_function()` - 100% identical
3. Feature importance calculation block - 98% identical
4. Gini calculation block - 100% identical
5. Performance logging - 100% identical
6. `model_performance()` - 100% identical

**Code Duplication:** ~700 lines
**Fix Effort:** 6 hours
**Expected Savings:** 650 lines

---

### CRITICAL - 3 Pipeline Classes (95% identical)
**File:** `/home/user/scoring_model/BOB_Scorecard_Training.ipynb`

**Classes:**
- `Pipeline_Base` (Cell 11)
- `Pipeline_Not_Good` (Cell 20)
- `Pipeline_Good` (Cell 26)

**Identical Methods:**
1. `train()` - 95% identical pipeline setup logic
2. `apply()` - 100% identical

**Code Duplication:** ~150 lines
**Fix Effort:** 4 hours
**Expected Savings:** 150 lines

---

### HIGH - Gini Formula (40+ occurrences)
**Pattern:** `2 * roc_auc_score(y, y_proba) - 1`

**Files with this pattern:**
- `/home/user/scoring_model/QNBAnalytics_ML/classifier.py` - 28 times
- `/home/user/scoring_model/QNBAnalytics_ML/elimination.py` - 5 times
- `/home/user/scoring_model/QNBAnalytics_ML/kpi.py` - 5 times
- `/home/user/scoring_model/QNBAnalytics_ML/kpi_calculator.py` - 2 times

**Code Duplication:** Formula scattered across 4 files
**Fix Effort:** 2 hours
**Expected Impact:** High (centralized metrics)

---

### HIGH - Pipeline Configuration (3 definitions)
**File:** `/home/user/scoring_model/BOB_Scorecard_Training.ipynb`

**Locations:**
- Cell 8: Base model config (apply_data_explore, apply_null_elimination, etc.)
- Cell 18: Layer 2 config (different values)
- Cell 26: Meta model config (different values)

**Problem:** Variables redefined 3 times with different values
**Fix Effort:** 2 hours
**Expected Savings:** 30 lines

---

### MEDIUM - Data Loading Logic (duplicated in 2 notebooks)
**Files:**
- `/home/user/scoring_model/BOB_Scorecard_Training.ipynb` (Cell 4)
- `/home/user/scoring_model/BOB_Scorecard_Scoring_Policy_Adjustment.ipynb` (Cell 3)

**Duplicated:**
- Database credential reading
- Database connection
- SQL file reading
- Data loading and preprocessing

**Code Duplication:** ~50 lines
**Fix Effort:** 3 hours
**Expected Savings:** 50 lines

---

### MEDIUM - Score Calculation Logic (3+ occurrences)
**Files:**
- `/home/user/scoring_model/BOB_Scorecard_Training.ipynb` (Cell 5, 27)
- `/home/user/scoring_model/BOB_Scorecard_Scoring_Policy_Adjustment.ipynb` (Cell 8)

**Pattern:**
```python
default_rate = np.where(default_probability == 0, 0.00001, default_probability)
odds = (1 / default_rate) - 1
scores = ((np.log(odds) - np.log(odds_at_ref)) / np.log(2)) * points_to_double + ref
```

**Code Duplication:** Formula repeated 3+ times
**Fix Effort:** 1 hour
**Expected Savings:** 20 lines

---

## Refactoring Checklist

### Phase 1: CRITICAL (Week 1)
- [ ] Create `BaseClassifier` in classifier.py
- [ ] Convert 4 model classes to subclasses of BaseClassifier
- [ ] Create `FlexiblePipeline` class in notebooks
- [ ] Replace 3 pipeline classes with single parameterized class
- [ ] Create `MetricsCalculator` utility with `calculate_gini()` method
- [ ] Update all gini calculations to use MetricsCalculator
- [ ] Test all models maintain same performance

### Phase 2: HIGH (Week 2)
- [ ] Create PIPELINE_CONFIGS dictionary at top of training notebook
- [ ] Replace 3 config sections with single config dictionary
- [ ] Create `DataLoader` class in QNBAnalytics_ML module
- [ ] Replace database/credential loading logic in both notebooks
- [ ] Create `CreditScoreCalculator` utility
- [ ] Replace score calculation logic in all locations

### Phase 3: OPTIONAL (Week 3)
- [ ] Consolidate feature importance logging
- [ ] Create shared model performance reporting
- [ ] Add comprehensive tests for refactored code
- [ ] Update documentation

---

## Quick Impact Summary

| Issue | Current State | After Refactoring | Improvement |
|-------|---------------|-------------------|-------------|
| Model Classes | 4 copies, ~1100 lines | 1 base + 4 subclasses, ~450 lines | 60% reduction |
| Pipeline Classes | 3 identical classes | 1 parameterized class | ~95% reduction |
| Gini Calculations | 40+ scattered | 1 utility method | 100% consolidation |
| Config Variables | 3 separate defs | 1 PIPELINE_CONFIGS dict | Centralized |
| Data Loading | 2 duplicates | 1 DataLoader class | Unified logic |
| Score Calculation | 3 duplicates | 1 CreditScoreCalculator | Consistent |
| **Total Code** | ~1500 dup. lines | 40% reduction | Maintainable |

---

## Success Metrics

**Before Refactoring:**
- Bug fix for model logic: 4 places to update
- Testing: 4 test suites for same logic
- Feature addition: Update 4 classes
- Configuration change: Update 3 locations

**After Refactoring:**
- Bug fix: 1 place to update (all models inherit fix)
- Testing: 1 test suite + 4 minimal subclass tests
- Feature addition: Update BaseClassifier + subclass overrides
- Configuration: Single PIPELINE_CONFIGS dictionary

**Expected Improvements:**
- Development speed: 3-4x faster for model changes
- Bug fix time: 4x faster
- Testing coverage: 40% fewer tests needed
- Maintenance: Significantly easier

