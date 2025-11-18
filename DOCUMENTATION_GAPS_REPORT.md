# DOCUMENTATION GAPS AND MISMATCHES - COMPREHENSIVE ANALYSIS
## QNBAnalytics ML - Credit Scoring System

**Project:** Bank of Baku Scorecard & Scoring Policy Adjustment  
**Codebase Size:** 4,899 lines of Python code | 13 core modules  
**Total Functions/Methods Identified:** 157  
**Functions WITH Docstrings:** 16 (10.2%)  
**Functions WITHOUT Docstrings:** 141 (89.8%)  
**Return Type Hints:** 2/157 (1.3%)  
**Parameter Type Hints:** <5 (minimal)  

---

## CRITICAL SEVERITY ISSUES

### 1. MISSING DOCSTRINGS (89.8% of functions)

#### File: `/home/user/scoring_model/QNBAnalytics_ML/binning.py`
**Impact:** HIGH - Core binning functionality undocumented

Missing Documentation:
- **Class `opt_binning`** (Lines 10-111): NO class docstring
  - `__init__()` - Parameters undocumented
  - `_validate_param_grid()` - No docstring, complex logic
  - `_find_column_params()` - Unclear return values
  - `fit()` - No documentation of what happens (lines 50-80)
  - `transform()` - Unclear output format (lines 82-96)
  - `get_binning_table()` - Returns 3 items but not documented (lines 98-111)

**Issue:** Users cannot understand how to use binning without reading code line-by-line.

---

#### File: `/home/user/scoring_model/QNBAnalytics_ML/data.py`
**Impact:** CRITICAL - Data loading is core functionality

Missing Documentation:
- **Function `connect_to_sql()`** (Lines 18-36): NO docstring
  - Parameters: username, password - not documented
  - Return type unclear (engine object)
  - No documentation of Oracle connection requirements
  
- **Function `data_load()`** (Lines 38-112): NO docstring
  - Complex conditional logic not explained
  - Handles both SQL and CSV/Excel - no clear documentation of behavior
  - Parameter `chunksize` undocumented
  - Returns pandas DataFrame but format/columns not documented
  - Throws generic exceptions - error types not documented

- **Function `train_test()`** (Lines 114-129): NO docstring
  - Parameters: split_style, group, stratify - unclear purpose
  - Return structure (train, test DataFrames) not documented
  - Hardcoded random_state=1 not explained

- **Function `explore()`** (Lines 131-186): NO docstring  
  - Complex statistical logic with hardcoded values
  - Lists like `categoric_types`, `numeric_types`, `date_types` - why these specific types?
  - Returns DataFrame with multiple columns - structure not documented
  - Role assignment logic ("ID", "target") undocumented

- **Function `export_to_sql()`** (Lines 188-197): NO docstring

- **Function `sample()`** (Lines 200-217): NO docstring
  - Parameter semantics unclear (0-1 = fraction, >1 = count)
  - Default behavior (n=0.1) undocumented

- **Function `merge()`** (Lines 220-250): NO docstring
  - Complex type handling (DataFrame, Series, ndarray)
  - Behavior varies by input type - not documented
  - Edge case in line 240: `merged_df = left.copy()` seems wrong

- **Function `conversion()`** (Lines 252-271): NO docstring

---

#### File: `/home/user/scoring_model/QNBAnalytics_ML/classifier.py`
**Impact:** CRITICAL - Model training logic completely undocumented

Missing Documentation:
- **Class `model_args`** (Lines 31-77): NO class docstring
  - `main_targeter()` - No docstring (lines 33-59)
    - Hardcoded threshold 0.01 for imbalanced detection - not explained
    - Returns two dicts with different structures - undocumented
  - `main_sampler()` - No docstring (lines 62-77)
    - Complex sampling logic with 6 return values - purpose unclear
    - Hardcoded 1/10 split for validation - no explanation

- **Class `LogisticRegression`** (Lines 80-310): Minimal documentation
  - `__init__()` - 15 parameters with NO documentation of purpose
  - `select_function()` - No docstring, undocumented formula (line 137)
  - `fit_model()` - 90+ lines of complex training logic with no overview
    - Creates model IDs from timestamps (lines 234-237) - why?
    - Rounding predictions to 38 decimals (line 251) - why 38?
  - `predict()` - No docstring for complex prediction logic
  - `model_performance()` - No explanation of return values

- **Class `RandomForest`** (Lines 313-539): Same issues as LogisticRegression

- **Class `LGBM`** (Lines 542-826): Same issues as LogisticRegression
  - **BUG:** Line 766 uses `self.model.feature_name_` (underscore-separated) 
    - Should be `self.model.feature_names_in_`
    - This will cause AttributeError at runtime

- **Class `XGBoost`** (Lines 829-1125): Same issues as LogisticRegression

---

#### File: `/home/user/scoring_model/QNBAnalytics_ML/preprocessing.py`
**Impact:** HIGH - Data preprocessing undocumented

Missing Documentation:
- **Function `remove_features()`** (Lines 11-13): NO docstring
  - **BUG:** Missing `inplace=True` - dataframe not actually modified
  
- **Class `conversions`** (Lines 15-44): NO class docstring
  - `to_numeric()` - No docstring
  - `to_categoric()` - No docstring
  - Hardcoded fill value (-987654321) - purpose undocumented

- **Class `impute_null`** (Lines 46-83): Minimal docstring in `__init__` only
  - `fit()`, `transform()`, `fit_transform()` - No docstrings

- **Class `encode`** (Lines 86-176): NO class docstring
  - `fit()`, `transform()`, `decoding()`, `decode()` - No docstrings
  - Complex decoding logic with undocumented woe_dict structure

- **Class `scale`** (Lines 178-212): NO class docstring
  - `fit()`, `transform()`, `fit_transform()` - No docstrings

---

#### File: `/home/user/scoring_model/QNBAnalytics_ML/elimination.py`
**Impact:** HIGH - Feature elimination logic has no documentation

Missing Documentation:
- **Function `_calculate_gini_table()`** (Lines 391-411): NO docstring
- **Function `_calculate_corr_table()`** (Lines 414-428): NO docstring
- **Function `_prepare_compare_table()`** (Lines 431-460): NO docstring
  - Hardcoded threshold 0.7 for correlation (line 433) - no explanation
- **Function `_eliminate_low_gini_correlated()`** (Lines 463-489): NO docstring
  - Variable names in Turkish (eleyen, elenen, kati_eleyen, etc.)
  - Complex logic with no explanation of algorithm
  - **Line 407:** Default value `threshold=0.7` inconsistent with other eliminators (usually 0.9)

- **Classes `DropLowGiniFeatures`, `DropNullFeatures`, `DropCorrelatedLowerGiniFeatures`**:
  - Have docstrings but parameters not fully documented
  - Line 141 docstring typo: "percetange" (should be "percentage")
  - Line 309 in docstring: "Notes: -" (incomplete)

---

#### File: `/home/user/scoring_model/QNBAnalytics_ML/feature_importances.py`
**Impact:** HIGH - SHAP calculation undocumented

Missing Documentation:
- **Function `_mean_shap_categoric_var()`** (Lines 15-33): NO docstring
- **Function `_shap_correlation()`** (Lines 35-47): NO docstring
- **Function `_raw_target_correlation()`** (Lines 50-51): NO docstring
- **Function `get_feature_importances()`** (Lines 85-180): Has docstring but incomplete
  - Complex conditional logic not explained
  - Line 150, 153, 160, 171: Comments reference SHAP bugs on GitHub but no explanation of workaround
  - Conditional import of shap module not documented (line 5 imports shap but sometimes fails?)

---

#### File: `/home/user/scoring_model/QNBAnalytics_ML/skills_library.py`
**Impact:** HIGH - 700+ lines with minimal documentation

Missing Documentation:
- **Class `DropCols`** (Lines 30-48): NO class docstring
  - `transform()` - Parameters 'y' and 'sample' not documented
  
- **Class `data_explore`** (Lines 51-98): NO class docstring
  - `fit()`, `transform()` - No docstrings
  - Hardcoded regex pattern `[^A-Za-z0-9_]+` not explained
  - Target correlation calculation (lines 68-76) not explained

- **Class `conversion`** (Lines 101-151): NO class docstring
  - `fit()`, `transform()` - No docstrings

- **Class `null_imputation`** (Lines 154-229): NO class docstring
  - Complex null handling logic undocumented
  - Line 181: `self.features_to_fix = np.array(...)` - why numpy array?
  
- **Class `encode_categoricals`** (Lines 231-272): NO class docstring
  - `fit()`, `transform()` - No docstrings

- **Class `scaling`** (Lines 275-311): NO class docstring
  - `fit()`, `transform()` - No docstrings

- **Class `FeatureElimination`** (Lines 314-380): NO class docstring
  - `_validate_eliminator()` - No docstring
  - Complex parameter mapping logic undocumented

- **Class `opt_binning`** (Lines 383-432): NO class docstring
  - `fit()`, `transform()`, `get_binning_table()` - No docstrings

- **Classes `model_LR`, `model_RF`, `model_LGBM`, `model_XGBoost`** (Lines 435-723):
  - All have identical structure with NO class docstrings
  - All `fit()` and `transform()` methods lack documentation
  - Model initialization parameters not documented
  - Duplicated code across 4 classes (maintenance risk)

---

#### File: `/home/user/scoring_model/QNBAnalytics_ML/kpi.py`
**Impact:** MEDIUM - KPI functions undocumented

Missing Documentation:
- **Function `kpi_calculator()`** (Lines 8-18): NO docstring
  - Parameters: varlist, x_train, y_train - purpose unclear
  - Return value (DataFrame) structure not documented
  - Hardcoded rounding to 4 decimals (line 13) - no explanation

- **Function `cv_score()`** (Lines 21-31): NO docstring
  - Hardcoded cv=5 (line 26) - no parameter for customization

---

#### File: `/home/user/scoring_model/QNBAnalytics_ML/kpi_calculator.py`
**Impact:** MEDIUM - Similar issues to kpi.py
- Missing module docstring
- No explanation of when to use `kpi_calculator` vs `cv_score`

---

#### File: `/home/user/scoring_model/QNBAnalytics_ML/pipelines.py`
**Impact:** MEDIUM - Pipeline management undocumented

Missing Documentation:
- **Class `model_pipeline`** (Lines 5-8): NO class docstring
  - Empty `__init__` with no explanation

- **Class `pipe_manager`** (Lines 13-117): NO class docstring
  - `register()` - No docstring (lines 16-63)
    - Complex logic building mdl_list with 10-element lists - structure not documented
    - File I/O operations undocumented
  - `apply_pipe()` - No docstring (lines 67-77)
  - `best_model()` - No docstring (lines 80-117)
    - Returns 4 values with unclear meaning

---

#### File: `/home/user/scoring_model/QNBAnalytics_ML/helpers.py`
**Impact:** LOW - Some documentation exists

Partially Documented:
- `getLogger()` - No docstring
- `getArgumentsList()` - No docstring
- `get_init_parameters()` - No docstring
- `all_logging_disabled()` - HAS docstring (lines 47-53)

---

#### File: `/home/user/scoring_model/QNBAnalytics_ML/skills_api.py`
**Impact:** CRITICAL - 1200+ line orchestration layer with minimal documentation

Documented:
- **Class `BasePipeline`** (Lines 118-193): Has docstring but incomplete
- Most methods in `BasePipeline` have docstrings (exceptions below)

Missing Documentation:
- **Decorator `variable_injection`** (Lines 15-115): NO docstring
  - Complex decorator with 100+ lines of undocumented logic
  - Parameter extraction and variable manipulation not explained
  
- **Function `get_feature_list()`** (Lines 15-27 in skills_library.py): NO docstring

- Several method parameters documented inconsistently
  - Example: `binning()` line 565 parameter descriptions incomplete
  - Line 588: "monotonic_trend*" marked with asterisk but not explained in notes

- **Class `ClassifierPipeline`** (Lines 795-1272): Has docstring but methods have issues
  - `LogisticRegression()` (lines 833-939): Docstring has typos
    - Line 858: "opzimization" (should be "optimization")
    - Line 884, 887: Copied docstring text about scaling (irrelevant for LogisticRegression)
  - Same docstring issues in `RandomForest()`, `LGBM()`, `XGBoost()` methods
  - Line 891: "gibi" (should be "gini") - copy-paste error

---

#### File: `/home/user/scoring_model/QNBAnalytics_ML/default_grids.py`
**Impact:** MEDIUM - Parameter grids undocumented

Issues:
- **NO module docstring** - purpose of file unclear
- **NO variable documentation** - what is "grids" dict used for?
- **Hardcoded hyperparameters** - why these specific values?
  - Line 7: C=[0.1, 0.3, 0.5, 1] for logistic regression - no explanation
  - Line 22-29: LightGBM parameters with no comments
- Different parameter keys across models not explained:
  - XGBoost uses 'early_stopping_rounds'
  - LightGBM uses 'early_stopping_round'
  - Why the difference?

---

#### File: `/home/user/scoring_model/QNBAnalytics_ML/param_grid_best.py`
**Impact:** LOW - Parameter reference file
- Has similar undocumented structure as default_grids.py
- Purpose of "base", "good", "not_good", "meta" grid names not explained

---

#### File: `/home/user/scoring_model/QNBAnalytics_ML/__init__.py`
**Impact:** LOW - Version file
- Only contains version string
- Should have module docstring

---

## CRITICAL CODE ISSUES (Not Just Documentation)

### 1. CONTRADICTORY CODE vs COMMENTS

**File: `binning.py` Line 3**
```python
### WOE BINNIG FUNCTIONS  # <-- TYPO: Should be "BINNING"
```
Impact: Suggests typo in code elsewhere

**File: `classifier.py` Line 244**
```python
logger.info("With ROC: " % roc_auc_score(y_train, self.train_prob))
# ^ Missing format string marker %s
```
Impact: Logging will print literally "With ROC: %" instead of actual value

**File: `classifier.py` Line 251 & 480 & 1066**
```python
self.test_pred_prob = np.round(self.test_pred_prob, 38)  # <-- WHY 38?
```
Impact: Rounding to 38 decimal places serves no purpose (floating point precision is ~15-17 digits)

**File: `classifier.py` Line 766**
```python
x_test = x_test[self.model.feature_name_]  # <-- WRONG ATTRIBUTE NAME
# Should be: self.model.feature_names_in_
```
Impact: RUNTIME ERROR - AttributeError will occur

**File: `preprocessing.py` Line 12**
```python
def remove_features(df, features, inplace):
    df.drop(columns=features)  # <-- Missing inplace=True parameter
    return df  # Returns original df, not modified!
```
Impact: Feature removal silently fails

**File: `elimination.py` Line 141**
```python
threshold: float, default=1
    The null threshold higher which a feature will be deemed to be uninformative...
# ^ Should be "higher than" not "higher which"
```

---

### 2. UNDOCUMENTED BUSINESS LOGIC & ASSUMPTIONS

#### Gini Score Calculation
**Impact:** Financial/credit decisions based on this metric

- Line 243: `self.train_gini = 2 * roc_auc_score(y_train, self.train_prob) - 1`
- Assumption: Gini = 2*AUC - 1
- Not documented why this specific formula
- Not explained for non-statistical users

#### Credit Risk Segmentation (From CODEBASE_ANALYSIS.md)
- 3-layer stacked ensemble mentioned in docs but implementation logic not clear
- Layer 1, 2, 3 separation not obvious in code

#### Hardcoded Constants (Not Explained)
- Random seed = 48 (why 48? Hard to find/change)
- Validation split = 1/10 (why not configurable?)
- Sample size defaults vary across modules (200000 vs 50000)
- Imputation fill value = -987654321 (sentinel value concept not documented)

#### Imbalance Detection Threshold
**File: `classifier.py` Line 52**
```python
if y_train.value_counts().min() / len(y_train) <= 0.01:
    # Uses class_weight='balanced' and scoring='f1'
# ^ Why 0.01 threshold? Not documented
```

#### Missing Null Documentation
- `drop_constant_features` function in elimination.py calls parent without passing all parameters
- `DropLowGiniFeatures` docstring mentions "drop correlated features" (copy-paste error)

---

## TYPE HINTS ANALYSIS

Only 2 functions have return type hints:

1. **`feature_importances.py` Line 53**
   ```python
   def _get_shap_feature_importances(...) -> pd.DataFrame:
   ```

2. **`feature_importances.py` Line 85**
   ```python
   def get_feature_importances(...) -> pd.DataFrame:
   ```

3. **`skills_api.py` Line 291** - Has partial type hints on parameters

**Missing Type Hints in:**
- ALL 4 model classes (LogisticRegression, RandomForest, LGBM, XGBoost)
- ALL preprocessing classes
- ALL elimination functions
- ALL data loading functions
- 99% of all other functions

**Impact:**
- IDE autocomplete doesn't work
- Type checking tools (mypy) cannot validate code
- Users must read source code to understand input/output types
- Harder to catch type-related bugs

---

## MISSING PARAMETER DOCUMENTATION

### Examples of Undocumented Parameters:

**`data.py` - `data_load()` function:**
```python
def data_load(engine = None, sql = None, data=None, index=None, 
              encoding='utf-8', chunksize=25000):
    # ^ encoding purpose unknown
    # ^ chunksize purpose unknown
    # ^ Why default 25000?
```

**`classifier.py` - `LogisticRegression.__init__()` (15 params):**
```python
def __init__(self,
             param_opt,              # Boolean but what does it enable?
             param_opt_method,       # String but what are valid values?
             train_size,             # Integer but relative to what?
             nprocs,                 # What does 'nprocs' stand for?
             random_state,           # Used for? Documented as 48 elsewhere
             refit_to_all,           # When to set to True/False?
             n_iter,                 # Iterations of what optimization?
             baseEstimator,          # What is this used for?
             param_grid,             # Format? Structure?
             cv,                     # CV strategy not explained
             verbose,                # Verbosity levels not documented
             alpha,                  # Formula for model selection not in __init__ docs
             time_limit,             # Not used anywhere - dead parameter?
             n_feature_params,       # Number of features to what?
             importance_direction_type,  # Only "shap" or "factor"? Not documented
             targetcorr_df           # What is structure?
             ):
```

**`skills_api.py` - `binning()` method (line 565):**
```python
def binning(self, method="optbinning", thresh=100000, nprocs=5, 
            max_n_bins=30, min_bin_size=0.0001, 
            monotonic_trend='auto_asc_desc', param_grid=None, 
            use_existing=False):
```

Documented Parameters: method, thresh, nprocs, max_n_bins, min_bin_size  
**Undocumented Parameters:** monotonic_trend (referenced in docs but not in method parameters section)  
**Use_existing Parameter:** Takes boolean or string path - behavior not explained

---

## MISSING RETURN VALUE DOCUMENTATION

**Most functions don't document return values:**

**`data.py` - `explore()` function (Returns complex DataFrame):**
```python
def explore(local_df):
    # Creates DataFrame with columns:
    # 'datatypes', 'role', 'count', 'unique_count', 'missing_rate', 
    # 'mean', 'min', 'max', 'std'
    # ^ NONE of this documented in function
    return summary
```

**`classifier.py` - `fit_model()` method:**
```python
def fit_model(self, x_train, y_train):
    # Creates attributes:
    # self.model, self.train_gini, self.cv_gini, self.feature_importances,
    # self.creation_, self.id_, self.train_time_
    # ^ Structure and format undocumented
```

**`elimination.py` - `_eliminate_low_gini_correlated()` function:**
```python
def _eliminate_low_gini_correlated(compare_table):
    # Returns a set
    # ^ Type of set elements not documented
    return set(kati_elenen)
```

**`skills_library.py` - `opt_binning.get_binning_table()` method:**
```python
def get_binning_table(self):
    return self.binning_tables, self.gini_table
    # ^ Structure of binning_tables, format of gini_table - undocumented
```

---

## MISSING MODULE-LEVEL DOCUMENTATION

### Files without module docstrings:
1. `binning.py` - Only has inline comment about "WOE BINNIG FUNCTIONS"
2. `default_grids.py` - No explanation of content
3. `preprocessing.py` - No module overview
4. `elimination.py` - No module overview
5. `feature_importances.py` - No module overview
6. `kpi.py` - No module overview
7. `kpi_calculator.py` - Same as kpi.py
8. `pipelines.py` - No module overview
9. `param_grid_best.py` - No module overview
10. `__init__.py` - Only version string

**Only documented modules:**
- `helpers.py` - Has docstring for function only, not module
- `data.py` - Has docstring for function only, not module
- `classifier.py` - Has docstring for function only, not module
- `skills_api.py` - Has class docstrings but no module overview
- `skills_library.py` - Has class docstrings but no module overview

---

## MISSING README / INCOMPLETE PROJECT DOCUMENTATION

### Existing Documentation:
- `CODEBASE_ANALYSIS.md` (21 KB) - Good architectural overview
- No comprehensive README.md at project root
- No INSTALLATION.md
- No USAGE.md or QUICKSTART.md
- No API_DOCUMENTATION.md

### What's Missing:
1. **README.md** - Project overview, installation, basic usage
2. **SETUP.md** - Environment setup, dependencies, database configuration
3. **USAGE_GUIDE.md** - How to train models, apply scoring, adjust policies
4. **API_REFERENCE.md** - Detailed function/class documentation
5. **ASSUMPTIONS.md** - Business logic, credit risk definitions, score interpretation
6. **TROUBLESHOOTING.md** - Common errors, solutions
7. **CONTRIBUTING.md** - Code style, documentation standards
8. **DATA_SCHEMA.md** - Input/output data formats and requirements

---

## IMPACT ASSESSMENT

### Severity: CRITICAL

**Maintainability:** 2/10
- New developers cannot understand code without reading every line
- High risk of breaking changes when modifying shared code
- Duplication across model classes (copy-paste maintenance trap)

**Usability:** 3/10
- Function parameters unclear
- Return value formats unknown
- Business logic assumptions undocumented
- Users forced to read implementation code

**Reliability:** 4/10
- Type checking impossible without type hints
- Runtime errors in LGBM.predict() (line 766)
- Silent failures in preprocessing.remove_features()
- Undocumented constants throughout

**Debuggability:** 2/10
- No function purpose documentation
- Complex logic without explanation
- Variables in Turkish language (elimination.py)
- Hardcoded values scattered throughout

---

## PRIORITY FIXES (By Impact & Effort)

### Priority 1 (CRITICAL - Do First):
1. Fix runtime bugs:
   - [ ] Line 766 in classifier.py: `feature_name_` → `feature_names_in_`
   - [ ] Line 12 in preprocessing.py: Add `inplace=True` to df.drop()
   - [ ] Line 244 in classifier.py: Fix logging format string
   
2. Add module docstrings to all Python files (15 minutes each)

3. Document key functions:
   - [ ] classifier.py: `model_args.main_targeter()`, `main_sampler()`
   - [ ] data.py: `data_load()`, `explore()`, `train_test()`
   - [ ] binning.py: `opt_binning` class and methods
   
4. Create API_DOCUMENTATION.md (4 hours)

### Priority 2 (HIGH - Do Next):
5. Add type hints to public functions (16 hours)
6. Document all undocumented methods in preprocessing.py (3 hours)
7. Document elimination.py functions (2 hours)
8. Create USAGE_GUIDE.md (3 hours)

### Priority 3 (MEDIUM - Schedule):
9. Create ASSUMPTIONS.md documenting business logic (3 hours)
10. Rename variables in elimination.py from Turkish to English (1 hour)
11. Create DATA_SCHEMA.md (2 hours)
12. Refactor duplicated model classes (8 hours)

---

## RECOMMENDATIONS

### 1. Implement Documentation Standards
- Require docstrings for all public functions/classes
- Use Google-style docstrings (Parameter, Returns, Raises sections)
- Add type hints to all new code
- Set up pre-commit hook to check documentation

### 2. Fix Critical Bugs
- Fix the 3 runtime/logic bugs identified above
- Add unit tests to catch similar issues

### 3. Improve Code Quality
- Extract duplicated code (4 model classes are identical)
- Rename variables to English (elimination.py)
- Document all hardcoded constants

### 4. Create Documentation
- Create README.md with quick start
- Create USAGE_GUIDE.md with examples
- Create API_DOCUMENTATION.md from docstrings (use Sphinx)

### 5. Tools to Implement
- **Pre-commit hooks** to enforce documentation
- **Sphinx** for documentation generation
- **mypy** for type checking
- **pydocstyle** for docstring consistency
- **darglint** to check docstring-code matching

---

## FILE SUMMARY TABLE

| File | Lines | Functions | With Docs | Without Docs | Missing Type Hints | Critical Issues |
|------|-------|-----------|-----------|--------------|-------------------|-----------------|
| classifier.py | 1129 | 29 | 0 | 29 | 29 | 1 (line 766) |
| skills_api.py | 1273 | 28 | 8 | 20 | 20 | 1 decorator |
| binning.py | 112 | 5 | 0 | 5 | 5 | 0 |
| data.py | 272 | 8 | 0 | 8 | 8 | 1 (line 240) |
| elimination.py | 598 | 11 | 3 | 8 | 8 | 1 (line 407) |
| feature_importances.py | 223 | 6 | 2 | 4 | 4 | 0 |
| preprocessing.py | 213 | 13 | 1 | 12 | 12 | 1 (line 12) |
| skills_library.py | 724 | 25 | 3 | 22 | 22 | 0 |
| helpers.py | 63 | 4 | 1 | 3 | 3 | 0 |
| pipelines.py | 118 | 6 | 0 | 6 | 6 | 0 |
| kpi.py | 32 | 2 | 0 | 2 | 2 | 0 |
| kpi_calculator.py | 32 | 2 | 0 | 2 | 2 | 0 |
| default_grids.py | 46 | 0 | 0 | 0 | 0 | 0 |
| param_grid_best.py | 21 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | **4,899** | **157** | **16** | **141** | **141** | **3-4** |

---

**Report Generated:** November 18, 2025
**Analysis Tool:** Claude Code File Analysis System
**Status:** Ready for Remediation

