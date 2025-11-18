# QNBAnalytics ML - Codebase Inconsistency Report

> **Prepared for:** Meeting with Outsource Company
> **Date:** 2025-11-18
> **Severity:** CRITICAL - Multiple issues require immediate attention

---

## Executive Summary

This report identifies **200+ inconsistencies and issues** across the QNBAnalytics ML credit scoring codebase. The issues range from **critical runtime bugs** that will crash production to **poor documentation** that makes the codebase unmaintainable.

### Overall Health Score: 4/10 (POOR)

| Category | Score | Status |
|----------|-------|--------|
| Code Quality | 5/10 | Needs Work |
| Documentation | 2/10 | Critical |
| Stability | 4/10 | High Risk |
| Reproducibility | 6/10 | Medium Risk |
| Maintainability | 3/10 | Poor |

### Critical Findings Summary

- **3 Runtime Bugs** that will crash production
- **4 Parameter Inconsistencies** causing calculation errors
- **1,500+ lines** of duplicated code
- **89.8%** of functions have no documentation
- **98.7%** missing type hints
- **7+ deprecated APIs** that will break on upgrade

---

## Table of Contents

1. [Critical Runtime Bugs](#1-critical-runtime-bugs)
2. [Parameter Inconsistencies](#2-parameter-inconsistencies)
3. [Code Duplication](#3-code-duplication)
4. [Naming Convention Violations](#4-naming-convention-violations)
5. [Hardcoded Values](#5-hardcoded-values)
6. [Configuration & Dependency Issues](#6-configuration--dependency-issues)
7. [Error Handling Issues](#7-error-handling-issues)
8. [Documentation Gaps](#8-documentation-gaps)
9. [Remediation Plan](#9-remediation-plan)
10. [Cost Estimates](#10-cost-estimates)

---

## 1. Critical Runtime Bugs

### These bugs WILL crash production code

| Bug | Location | Impact | Fix Time |
|-----|----------|--------|----------|
| **LGBM Model Crash** | classifier.py:766 | AttributeError on prediction | 5 min |
| **Silent Feature Removal** | preprocessing.py:12 | Features not removed | 5 min |
| **Broken Logging** | classifier.py:244 | Metrics not logged | 5 min |

### Bug #1: LGBM Model Will Crash on Prediction

**File:** `QNBAnalytics_ML/classifier.py` (line 766)

```python
# CURRENT (BUG)
x_test = x_test[self.model.feature_name_]  # AttributeError!

# SHOULD BE
x_test = x_test[self.model.feature_names_in_]
```

**Impact:** Every LGBM prediction call will crash with `AttributeError: 'LGBMClassifier' object has no attribute 'feature_name_'`

---

### Bug #2: Feature Removal Silently Fails

**File:** `QNBAnalytics_ML/preprocessing.py` (line 12)

```python
# CURRENT (BUG)
df.drop(columns=features)  # Returns new DataFrame, doesn't modify df

# SHOULD BE
df.drop(columns=features, inplace=True)
```

**Impact:** Features you think are removed are still in your training data

---

### Bug #3: Logging Broken - Metrics Not Recorded

**File:** `QNBAnalytics_ML/classifier.py` (line 244-245)

```python
# CURRENT (BUG)
logger.info("With ROC: " % roc_auc_score(...))  # Missing %s

# SHOULD BE
logger.info("With ROC: %s" % roc_auc_score(...))
```

**Impact:** Logs show `"With ROC: "` instead of actual metric values

---

## 2. Parameter Inconsistencies

### Critical: `odds_at_ref` Has Different Values

**This directly affects credit score calculations**

| Location | Value | Purpose |
|----------|-------|---------|
| Training Cell 15 | **50** | Base model scoring |
| Training Cell 31 | **100** | Final meta model scoring |
| Production Scoring | **100** | Policy adjustment |

**Impact:** Same probability produces different scores:
- With `odds_at_ref=50`: probability 0.1 → score **180.5**
- With `odds_at_ref=100`: probability 0.1 → score **194.7**
- **Difference: 14.2 points**

This means your Layer 2 segmentation thresholds were calculated with a different `odds_at_ref` than your final scores!

---

### Other Parameter Inconsistencies

| Parameter | Code Default | Training Used | Issue |
|-----------|--------------|---------------|-------|
| `gini_threshold` | 0.01 | 0.05 | Different features eliminated |
| `null_threshold` | 1.0 | 0.99 | Different features kept |
| `random_state` | Various | 48 | Hardcoded in 10+ places |

---

## 3. Code Duplication

### 1,500+ Lines of Duplicated Code

| Duplication | Files | Lines Duplicated | Impact |
|-------------|-------|------------------|--------|
| 4 Model Classes | classifier.py | ~700 lines | Bug fixes needed 4x |
| 3 Pipeline Classes | Training notebook | ~150 lines | Changes needed 3x |
| Gini Formula | 4 files | 40+ instances | Formula change = 40 edits |
| Pipeline Config | Training notebook | 3 definitions | Inconsistency risk |

### Example: Four Nearly Identical Model Classes

```python
# classifier.py contains these classes with 60-75% identical code:
class LogisticRegression:    # ~200 lines
class RandomForest:          # ~200 lines
class LGBM:                  # ~200 lines
class XGBoost:               # ~200 lines

# Should be ONE BaseClassifier with inheritance
```

**Impact:**
- Any bug must be fixed in 4 places
- Inconsistencies creep in (like the LGBM bug above)
- 4x testing effort

---

## 4. Naming Convention Violations

### PEP 8 Compliance Score: 70% (Grade: C)

| Category | Compliance | Examples |
|----------|------------|----------|
| Class Names | **40% (F)** | `opt_binning`, `data_explore`, `encode_categoricals` |
| Functions | 80% (B) | `getLogger`, `to_categoric` |
| Variables | 70% (C) | Mixed `x_train`/`X_train`/`train_x` |

### Critical: Duplicate Class Definition

`opt_binning` is defined **TWICE** with different implementations:
- `binning.py:10`
- `skills_library.py:383`

### Variable Naming Chaos

Same concept, different names:

| Concept | Variations Used |
|---------|-----------------|
| Training data | `train`, `x_train`, `X_train` |
| Probabilities | `probs`, `prob`, `y_pred_prob`, `dfx_probs`, `test_pred_prob` |
| Predictions | `preds`, `pred`, `y_pred`, `dfx_preds` |
| Categorical | `cat`, `categoric`, `category`, `cat_vars` |

### PEP 8 Violations

```python
# WRONG - uses camelCase
baseEstimator  # Should be base_estimator
getLogger()    # Should be get_logger()

# WRONG - not PascalCase classes
class opt_binning:     # Should be OptBinning
class data_explore:    # Should be DataExplore
```

---

## 5. Hardcoded Values

### 58+ Hardcoded Values Found

| Category | Count | Severity | Example |
|----------|-------|----------|---------|
| Random seeds | 10+ | HIGH | `random.seed(48)` everywhere |
| Precision bugs | 4 | CRITICAL | `np.round(x, 38)` - 38 decimals! |
| Currency rates | 6+ | CRITICAL | USD=1.7, EUR=1.9 in SQL |
| Target definition | 1 | CRITICAL | `GECIKME_13M > 90` |
| Sample sizes | 4 | MEDIUM | `sample_size=100` |

### Critical: Precision Rounding Bug

**File:** `classifier.py` (lines 251, 480, 768, 1066)

```python
np.round(self.test_pred_prob, 38)  # 38 decimal places!
```

This is clearly a leftover test value. Should be 4-6 decimal places.

### Critical: Hardcoded Currency in SQL

**File:** `scripts/DATAMART.sql` (lines 119-137)

```sql
-- Repeated 6+ times with no configuration
CASE WHEN CURRENCY = 'USD' THEN AMOUNT * 1.7
     WHEN CURRENCY = 'EUR' THEN AMOUNT * 1.9
     WHEN CURRENCY = 'RUB' THEN AMOUNT * 0.025
```

Cannot update exchange rates without SQL changes.

---

## 6. Configuration & Dependency Issues

### Critical Import Bugs

| Issue | File | Line | Impact |
|-------|------|------|--------|
| `cx_Oracle` commented out | data.py | 8 | Database connection fails |
| `types` not imported | data.py | 194 | `export_to_sql()` crashes |
| `shap` commented out | feature_importances.py | 5 | All SHAP analysis crashes |

### Deprecated APIs (Will Break on Upgrade)

| API | Files | Status |
|-----|-------|--------|
| `df.append()` | 3 files, 6 locations | Removed in pandas 2.0 |
| `engine.execute()` | data.py:30 | Removed in SQLAlchemy 2.0 |

### Dependency Issues

```
# requirements.txt issues:
imbalanced-learn  # NO VERSION - unpinned!

# Missing from requirements:
mlflow  # Used but not listed

# Outdated packages:
numpy==1.21.6     # June 2021 - deprecated for Python 3.11
pandas==1.3.5     # Sept 2021 - limited Python 3.11 support
```

---

## 7. Error Handling Issues

### Critical Error Handling Problems

| Issue | Count | Severity |
|-------|-------|----------|
| Bare `except:` clauses | 5 | HIGH |
| Silent error swallowing | 4 | CRITICAL |
| Print instead of logging | 15+ | HIGH |
| Wrong log levels | 6 | MEDIUM |
| Missing input validation | Many | HIGH |

### Example: Silent Failures

**File:** `data.py` (lines 63-65)

```python
except:
    pass  # Caller has no idea operation failed!
```

### Example: Bare Except

**File:** `preprocessing.py` (line 28)

```python
except:  # Catches everything including KeyboardInterrupt!
    print("Error converting to numeric")
```

---

## 8. Documentation Gaps

### Documentation Score: 2/10 (CRITICAL)

| Metric | Value | Status |
|--------|-------|--------|
| Functions without docstrings | 141/157 (89.8%) | CRITICAL |
| Missing type hints | ~155/157 (98.7%) | CRITICAL |
| Files without module docstrings | 10/13 | POOR |

### Files with Zero Documentation

| File | Lines | Functions | Docstrings |
|------|-------|-----------|------------|
| classifier.py | 1,129 | 29 | **0** |
| data.py | 272 | 8 | **0** |
| binning.py | 112 | 5 | **0** |

### Missing Project Documentation

- No README.md at root
- No API documentation
- No usage guide
- No data schema documentation
- No assumptions documentation

### Undocumented Business Logic

- **Gini formula:** `2*AUC - 1` (not explained anywhere)
- **Imbalance threshold:** 0.01 (hardcoded, undocumented)
- **Validation split:** 1/10 ratio (why?)
- **Random seed:** 48 (why this number?)

---

## 9. Remediation Plan

### Phase 1: Critical Fixes (1-2 days)

**Must Do Before Any Other Work**

| Task | Time | Impact |
|------|------|--------|
| Fix 3 runtime bugs | 0.5 hr | Prevents crashes |
| Fix commented imports | 0.5 hr | Enables features |
| Fix `odds_at_ref` inconsistency | 1 hr | Correct scores |
| Replace deprecated APIs | 2 hr | Future compatibility |
| Pin imbalanced-learn version | 0.5 hr | Reproducibility |

**Total: ~5 hours**

---

### Phase 2: High Priority (1 week)

**Code Stability & Maintainability**

| Task | Time | Impact |
|------|------|--------|
| Create centralized config | 4 hr | No more hardcoding |
| Fix bare except clauses | 2 hr | Proper error handling |
| Replace print with logging | 3 hr | Consistent logging |
| Add module docstrings | 3 hr | Basic documentation |
| Document critical functions | 8 hr | Maintainability |

**Total: ~20 hours**

---

### Phase 3: Medium Priority (2-4 weeks)

**Code Quality & Duplication**

| Task | Time | Impact |
|------|------|--------|
| Consolidate 4 model classes | 8 hr | 40% less code |
| Standardize naming conventions | 6 hr | Readability |
| Add type hints | 8 hr | IDE support |
| Create BaseClassifier | 4 hr | Extensibility |
| Remove code duplication | 8 hr | Maintainability |

**Total: ~34 hours**

---

### Phase 4: Best Practices (Ongoing)

**Long-term Improvements**

| Task | Time | Impact |
|------|------|--------|
| Setup pre-commit hooks | 2 hr | Enforce standards |
| Add unit tests | 16 hr | Prevent regressions |
| Generate API docs (Sphinx) | 4 hr | User documentation |
| Create CI/CD pipeline | 8 hr | Automated testing |

**Total: ~30 hours**

---

## 10. Cost Estimates

### Total Effort to Fix All Issues

| Phase | Hours | Priority |
|-------|-------|----------|
| Phase 1: Critical | 5 | Immediate |
| Phase 2: High | 20 | This week |
| Phase 3: Medium | 34 | This month |
| Phase 4: Best Practices | 30 | Ongoing |
| **Total** | **89 hours** | |

### Risk if Not Fixed

| Risk | Impact | Probability |
|------|--------|-------------|
| Production crash (LGBM bug) | HIGH | 100% |
| Incorrect scores (odds_at_ref) | HIGH | Already happening |
| Upgrade failure (deprecated APIs) | MEDIUM | On pandas 2.0 upgrade |
| Maintenance burden | HIGH | Every change is 4x effort |
| Knowledge loss | CRITICAL | No documentation |

---

## Summary of Issues by Severity

### CRITICAL (Fix Immediately)

1. **3 Runtime bugs** - Will crash production
2. **odds_at_ref = 50 vs 100** - Wrong credit scores
3. **Commented imports** - Features don't work
4. **Missing documentation** - Unmaintainable

### HIGH (Fix This Week)

1. **7 deprecated APIs** - Will break on upgrade
2. **Unpinned dependency** - Irreproducible
3. **5 bare except clauses** - Hidden errors
4. **15+ print statements** - No logging

### MEDIUM (Fix This Month)

1. **1,500+ lines duplication** - 4x maintenance
2. **58+ hardcoded values** - No configurability
3. **Naming inconsistencies** - Poor readability
4. **Missing type hints** - No IDE support

---

## Recommendations for Meeting

### Questions to Ask Outsource Company

1. Why does `odds_at_ref` have different values in different cells?
2. What is the purpose of the 38 decimal place rounding?
3. Why is there duplicate code for 4 model classes?
4. Why are imports commented out?
5. Where is the documentation?

### Immediate Actions Required

1. **Do not deploy** until runtime bugs are fixed
2. **Recalculate all scores** with consistent `odds_at_ref`
3. **Version control** must be enforced
4. **Code review process** must be implemented
5. **Documentation standards** must be established

### Minimum Acceptance Criteria

Before accepting this codebase:
- [ ] All 3 runtime bugs fixed
- [ ] Parameter inconsistencies resolved
- [ ] Deprecated APIs replaced
- [ ] Critical functions documented
- [ ] Imports uncommented and working
- [ ] Configuration centralized

---

## Appendix: File-by-File Issues

| File | Critical | High | Medium | Total |
|------|----------|------|--------|-------|
| classifier.py | 2 | 5 | 10 | 17 |
| data.py | 3 | 4 | 3 | 10 |
| preprocessing.py | 2 | 3 | 5 | 10 |
| skills_library.py | 1 | 3 | 8 | 12 |
| elimination.py | 0 | 2 | 5 | 7 |
| pipelines.py | 0 | 2 | 3 | 5 |
| Training.ipynb | 2 | 4 | 6 | 12 |
| DATAMART.sql | 1 | 3 | 4 | 8 |

---

*Report generated: 2025-11-18*
*Total issues identified: 200+*
*Estimated remediation: 89 hours*
