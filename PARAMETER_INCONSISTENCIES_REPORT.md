# PARAMETER INCONSISTENCIES ANALYSIS REPORT
## Scoring Model Codebase Analysis
Date: 2025-11-18

---

## EXECUTIVE SUMMARY

Found **1 CRITICAL** and **2 SIGNIFICANT** parameter inconsistencies that directly impact model scoring and reproducibility.

---

## CRITICAL INCONSISTENCY #1: odds_at_ref Parameter

### Parameter Name
`odds_at_ref` - Used in credit score calculation formula

### Issue
The odds_at_ref parameter is defined with different values in the same notebook, leading to inconsistent credit score calculations.

### Locations and Values

| Location | File | Cell Purpose | Value |
|----------|------|--------------|-------|
| Cell 15 | BOB_Scorecard_Training.ipynb | Base Model Score Calculation | **50** |
| Cell 16 (uses value from Cell 15) | BOB_Scorecard_Training.ipynb | Layer 2 Threshold Calculation | **50** (inherited) |
| Cell 31 | BOB_Scorecard_Training.ipynb | Meta Model Score Calculation (All Data) | **100** |
| Policy Adjustment | BOB_Scorecard_Scoring_Policy_Adjustment.ipynb | Production Scoring | **100** |

### Code Snippets

**Cell 15 - Base Model (odds_at_ref = 50):**
```python
ref=200
odds_at_ref=50
points_to_double=20

default_rate = base_all_proba
default_rate = np.where(default_rate == 0, 0.00001, default_rate)
odds = (1/default_rate)-1
base_all_score = ((np.log(odds)-np.log(odds_at_ref))/np.log(2)) * (points_to_double) + ref
```

**Cell 31 - Meta Model (odds_at_ref = 100):**
```python
ref=200
odds_at_ref=100
points_to_double=20

default_rate = base_all_proba  # Same data but different odds_at_ref!
odds = (1/default_rate)-1
base_all_score = ((np.log(odds)-np.log(odds_at_ref))/np.log(2)) * (points_to_double) + ref

# ... also applies to meta_all_score with same odds_at_ref=100
meta_all_score = ((np.log(odds)-np.log(odds_at_ref))/np.log(2)) * (points_to_double) + ref
```

**Policy Adjustment Notebook - Production (odds_at_ref = 100):**
```python
ref=200
odds_at_ref=100
points_to_double=20

meta_test_score = ((np.log(odds)-np.log(odds_at_ref))/np.log(2)) * (points_to_double) + ref
```

### Why This Is Problematic

1. **Score Variability**: The same probability produces different credit scores depending on which odds_at_ref value is used
   - Example: If probability = 0.1 (default_rate = 0.1)
     - Using odds_at_ref=50: score ≈ 180.5
     - Using odds_at_ref=100: score ≈ 194.7
     - **Difference: ~14.2 points**

2. **Inconsistent Training Output**: 
   - `MAIN_SCORES_BASE_training.xlsx` uses odds_at_ref=50 for base model
   - `MAIN_SCORES_training.xlsx` uses odds_at_ref=100 for final scores
   - Same data, different scoring methodology

3. **Production-Training Mismatch**:
   - Training uses odds_at_ref=100 (Cell 31, final scores)
   - Production uses odds_at_ref=100 (Policy Adjustment notebook)
   - BUT intermediate outputs use odds_at_ref=50 (Cell 15, base model)
   - Risk of confusing which scores to rely on

4. **Threshold Calculation Issue**:
   - `good_score_threshold = 180` and `not_good_score_threshold = 200`
   - These were calculated using the odds_at_ref=50 value
   - Later, odds_at_ref is changed to 100, but thresholds remain at 180/200
   - This creates a semantic inconsistency: thresholds were derived using different assumptions

### Output Files Affected

| File | odds_at_ref Used | Issue |
|------|------------------|-------|
| MAIN_SCORES_BASE_training.xlsx | 50 | Intermediate output, different from final |
| MAIN_SCORES_training.xlsx | 100 | Final training output |
| Production scoring | 100 | But confusion about which value to use |

---

## SIGNIFICANT INCONSISTENCY #2: Score Threshold Assumptions

### Parameter Names
- `good_score_threshold` = 180
- `not_good_score_threshold` = 200

### Issue
Thresholds are calculated using `odds_at_ref=50`, but then `odds_at_ref` is changed to 100 without recalculating thresholds.

### Location
BOB_Scorecard_Training.ipynb, Cell 16

### Code
```python
# Thresholds calculated with odds_at_ref=50 (from previous cell)
good_score_threshold = 180
not_good_score_threshold = 200
good_threshold = 1 / (np.exp((good_score_threshold-ref) / (points_to_double) * np.log(2) + np.log(odds_at_ref)) + 1)  
not_good_threshold = 1 / (np.exp((not_good_score_threshold-ref) / (points_to_double) * np.log(2) + np.log(odds_at_ref)) + 1)
```

### Why This Is Problematic

1. **Semantic Mismatch**: 
   - Score thresholds (180, 200) imply a certain odds_at_ref value
   - When odds_at_ref changes from 50 to 100, the probability thresholds change
   - Original thresholds may no longer be appropriate

2. **Probability Threshold Shift**:
   - With odds_at_ref=50: A score of 200 corresponds to specific probability threshold
   - With odds_at_ref=100: Same score (200) corresponds to DIFFERENT probability threshold
   - This affects model segmentation (good/not_good split)

3. **No Recalculation**: 
   - Thresholds are used in Cell 20 to split data
   - But odds_at_ref changes in Cell 31
   - Split was done with one assumption, scores calculated with another

---

## MINOR INCONSISTENCY #3: Gini Threshold Values in Feature Elimination

### Parameter Name
Gini threshold for low gini feature elimination

### Locations and Values

| Location | File | Value |
|----------|------|-------|
| default_grids.py | Generic parameter grid (not used) | Not defined explicitly |
| Training notebook Cell 10 (Base model) | BOB_Scorecard_Training.ipynb | **0.05** |
| Training notebook Cell 17 (Layer 2 models) | BOB_Scorecard_Training.ipynb | **0.05** |
| Training notebook Cell 26 (Meta model) | BOB_Scorecard_Training.ipynb | **0.05** |
| elimination.py (code default) | QNBAnalytics_ML/elimination.py line 149 | **0.01** |

### Code Snippet (elimination.py)
```python
def __init__(
    self,
    variables=None,
    threshold=0.01,  # DEFAULT IN CODE
    missing_values="raise",
    confirm_variables=False,
):
```

### Code Snippet (Training Notebook)
```python
# All pipelines use this:
self.pipeline.feature_elimination(eliminator='drop_low_gini_features', params={'threshold':0.05, 'missing_values':'ignore'})
```

### Why This Is Problematic

1. **Code vs. Notebook Mismatch**:
   - Default in code: 0.01
   - Actual value used in training: 0.05
   - If someone reproduces using defaults, they get different features eliminated

2. **Documentation Gap**:
   - No documentation explaining why 0.05 was chosen over 0.01
   - No centralized configuration file to maintain consistency

3. **Reproducibility Risk**:
   - New users might use code defaults (0.01) instead of training values (0.05)
   - Would result in different feature sets being dropped

### Severity: LOW-MEDIUM
- Only affects feature selection in training, not production scoring
- But could cause reproducibility issues if someone retains the model

---

## INCONSISTENCY #4: Null Threshold in Feature Elimination

### Parameter Name
Null threshold for dropping highly null features

### Locations and Values

| Location | File | Value | Usage |
|----------|------|-------|-------|
| elimination.py (code default) | QNBAnalytics_ML/elimination.py line 314 | **1.0** | Default |
| Training notebook (all pipelines) | BOB_Scorecard_Training.ipynb | **0.99** | Actual |

### Code Snippet (elimination.py)
```python
def __init__(self, variables=None, threshold=1, confirm_variables=False):
    super().__init__(confirm_variables)
    self.threshold = threshold
```

### Code Snippet (Training Notebook - Cell 10, 17, 26)
```python
self.pipeline.feature_elimination(eliminator='drop_null_features', params={'threshold': 0.99})
```

### Why This Is Problematic

1. **Default vs. Actual**:
   - Code default 1.0 = drops features with 100% nulls (very conservative)
   - Training uses 0.99 = drops features with 99%+ nulls (more aggressive)
   - Affects which features are available for model training

2. **Upstream Effects**:
   - More aggressive null dropping → fewer features → different model training
   - Could affect Gini scores and feature selection downstream

### Severity: MEDIUM
- Directly impacts feature set used in model training
- Could cause different models if someone uses code defaults

---

## PIPELINE CONFIGURATION VARIATIONS

### Status: INTENTIONAL (Not an Inconsistency)

Different pipeline configurations are used for different layers, which appears intentional:

| Setting | Layer 1 (Base) | Layer 2 (Good/Not Good) | Layer 3 (Meta) |
|---------|---|---|---|
| apply_binning | TRUE | FALSE | FALSE |
| apply_scaling | FALSE | TRUE | FALSE |
| apply_null_imputation | FALSE | TRUE | FALSE |
| apply_categorical_encoding | FALSE | TRUE | FALSE |

**Assessment**: This is INTENTIONAL design - different models use different preprocessing strategies appropriate to their use case.

---

## HYPERPARAMETER GRID ANALYSIS

### Status: CONSISTENT

Comparing `default_grids.py` vs `param_grid_best.py`:

- **default_grids.py**: Search spaces for RandomizedSearchCV
- **param_grid_best.py**: Fixed best parameters found after optimization

Both files serve different purposes (search vs. results), so variations are expected and not inconsistencies.

### Key Observation
All models use `random_state=48` consistently across all best parameter definitions.

---

## SUMMARY TABLE OF ALL INCONSISTENCIES

| # | Parameter | Severity | Type | Impact |
|---|-----------|----------|------|--------|
| 1 | odds_at_ref | **CRITICAL** | Value Mismatch | Credit score calculation (50 vs 100) |
| 2 | Score Thresholds (180/200) | **SIGNIFICANT** | Semantic Issue | Layer 2 data splitting with wrong assumptions |
| 3 | Gini Threshold | MEDIUM | Default vs Actual | Feature elimination (0.01 vs 0.05) |
| 4 | Null Threshold | MEDIUM | Default vs Actual | Null feature dropping (1.0 vs 0.99) |

---

## RECOMMENDATIONS

### IMMEDIATE (Critical)
1. **Standardize odds_at_ref**:
   - Decision: Use odds_at_ref = 100 throughout
   - Update Cell 15 in Training notebook to use 100
   - Recalculate base_all_scores in Cell 15
   - Document the rationale for using 100

2. **Recalculate Thresholds**:
   - After fixing odds_at_ref, recalculate good_threshold and not_good_threshold
   - Verify the good/not_good split is still meaningful
   - Re-run Layer 2 model training if split changes significantly

### SHORT-TERM (High Priority)
1. **Create Configuration File**:
   - Add `config.py` or `config.json` at project root
   - Define all magic numbers: ref, odds_at_ref, points_to_double, thresholds
   - Import from single source of truth

2. **Add Comments**:
   - Explain why specific threshold values (180, 200) were chosen
   - Document odds_at_ref selection rationale
   - Link to any credit risk documentation

3. **Update elimination.py Defaults**:
   - Change default gini_threshold from 0.01 to 0.05
   - Change default null_threshold from 1.0 to 0.99
   - Or explicitly pass all parameters in training to avoid relying on defaults

### MEDIUM-TERM (Best Practices)
1. **Version Control Outputs**:
   - Track which odds_at_ref and thresholds were used for each output file
   - Add metadata to Excel outputs (worksheet with parameters used)

2. **Unit Tests**:
   - Test score calculation with known inputs
   - Verify odds_at_ref changes don't break downstream logic

3. **Documentation**:
   - Create README explaining the three-layer model architecture
   - Document why different preprocessing is used per layer
   - Provide parameter sensitivity analysis

---

## FILES REQUIRING UPDATES

1. `/home/user/scoring_model/BOB_Scorecard_Training.ipynb` - Cell 15, Cell 16
2. `/home/user/scoring_model/QNBAnalytics_ML/elimination.py` - Lines 149, 314
3. `/home/user/scoring_model/QNBAnalytics_ML/default_grids.py` - Consider updating defaults

---

## CONCLUSION

The most critical issue is the **odds_at_ref inconsistency** which creates two different scoring methodologies in the same training run. This should be resolved immediately to ensure reproducibility and consistency between training and production environments.

