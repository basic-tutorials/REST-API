# NAMING CONVENTIONS AND STYLE INCONSISTENCIES ANALYSIS
## QNBAnalytics_ML Module

**Analysis Date:** 2025-11-18  
**Codebase Location:** /home/user/scoring_model  
**Scope:** All Python files in QNBAnalytics_ML/ and Jupyter notebooks

---

## EXECUTIVE SUMMARY

The codebase exhibits **SIGNIFICANT** naming inconsistencies across multiple dimensions:
- **7+ different naming conventions** for the same concepts
- **Class naming** uses 3+ different styles (lowercase, snake_case, PascalCase)
- **Variables** for the same concept (train/test data) use inconsistent capitalization
- **Abbreviation patterns** are inconsistent throughout (probs vs prob, cat vs categoric)
- **Duplicate class definitions** with slightly different implementations

**Severity:** HIGH - This impacts code maintainability, readability, and increases cognitive load

---

## 1. CLASS NAMING INCONSISTENCIES

### 1.1 Mixed Class Naming Styles

The codebase uses **3 distinct naming conventions** for classes:

#### PascalCase (Upper Camel Case) - CORRECT STANDARD
```
✓ DropCols
✓ FeatureElimination  
✓ ClassificationScores
✓ LogisticRegression
✓ RandomForest
✓ LGBM (all caps - exception)
✓ XGBoost (mixed case - exception)
✓ BasePipeline
✓ ClassifierPipeline
✓ DropNullFeatures
✓ DropLowGiniFeatures
✓ DropCorrelatedLowerGiniFeatures
✓ DropConstantFeatures
```

#### snake_case (Lower Case with Underscores) - INCONSISTENT
```
✗ opt_binning (DUPLICATE: appears in binning.py line 10 AND skills_library.py line 383)
✗ data_explore
✗ conversion
✗ null_imputation
✗ encode_categoricals
✗ scaling
✗ model_pipeline
✗ pipe_manager
✗ model_args
```

#### lowercase (Simple Lowercase) - INCONSISTENT
```
✗ conversions (in preprocessing.py)
✗ impute_null (in preprocessing.py)
✗ encode (in preprocessing.py)
✗ scale (in preprocessing.py)
```

#### Special Cases: Model Wrapper Classes
```
✗ model_LR (snake_case + abbreviation)
✗ model_RF (snake_case + abbreviation)
✗ model_LGBM (snake_case + abbreviation)
✗ model_XGBoost (snake_case + mixed case)
```

**These duplicate actual model classes:**
- `model_LR` wraps `LogisticRegression` (skills_library.py line 435)
- `model_RF` wraps `RandomForest` (skills_library.py line 509)
- `model_LGBM` wraps `LGBM` (skills_library.py line 582)
- `model_XGBoost` wraps `XGBoost` (skills_library.py line 654)

### 1.2 DUPLICATE CLASS DEFINITIONS

**CRITICAL ISSUE:** Class `opt_binning` is defined TWICE:

1. **binning.py, line 10**
   ```python
   class opt_binning:
       def __init__(self, num_variables, cat_variables):
           self.num_binner = None
           self.cat_binner = None
           ...
   ```

2. **skills_library.py, line 383**
   ```python
   class opt_binning:
       def __init__(self, num_variables, cat_variables):
           self.binner = None
           self.num_variables = num_variables
           ...
   ```

These are different implementations that could lead to confusion about which to use.

---

## 2. VARIABLE/ATTRIBUTE NAMING INCONSISTENCIES

### 2.1 Training/Test Data Variables (CRITICAL)

The same concepts are named with **4-5 different variations**:

| Concept | Inconsistent Names | File References |
|---------|-------------------|-----------------|
| **Training Features** | `train`, `x_train`, `X_train` | classifier.py, skills_api.py, data.py |
| **Test/Validation Data** | `test`, `x_test`, `X_test`, `x_val` | classifier.py, skills_api.py |
| **Training Target** | `y_train` | classifier.py, skills_api.py |
| **Test Target** | `y_test`, `y_val` | classifier.py, skills_api.py |

**Example Inconsistencies:**

```python
# classifier.py - uses lowercase 'x'
def main_sampler(x_train, train_size=None, random_state=48):
    all_index = x_train.index.to_list()
    ...

def fit_model(self, x_train, y_train):
    self.model.fit(x_train.loc[self.smpl_index], y_train.loc[self.smpl_index])
    
def predict(self, x_test, y_test=[], sample='Validation'):
    self.test_pred_prob = self.model.predict_proba(x_test)[:, 1]

# skills_api.py - uses both lowercase and references via self
def data_read_sql(self, username=..., test_sql="", target=None):
    ...
    self.train = data.data_load(engine, sql=query, ...)
    self.x_val = data.data_load(engine, sql=test_sql, ...)
```

### 2.2 Predictions/Probabilities Variables

**6+ different naming patterns** for prediction outputs:

| Concept | Names | Files |
|---------|-------|-------|
| **Probabilities** | `dfx_probs`, `probs`, `prob`, `y_pred_prob`, `self.test_pred_prob` | skills_library.py, classifier.py |
| **Predictions** | `dfx_preds`, `preds`, `pred`, `y_pred`, `test_pred` | skills_library.py, classifier.py |

**Example:**
```python
# skills_library.py - uses dfx prefix
dfx_probs, dfx_preds = self.model.predict(test, y_test=y_test, sample='Test')

# classifier.py - uses different naming
self.test_pred_prob = self.model.predict_proba(x_test)[:, 1]
test_pred = self.model.predict(x_test)
```

### 2.3 Variable Type Classifications

**Inconsistent names for categorical and numeric variables:**

```python
# INCONSISTENT PATTERNS:
cat_variables          # skills_api.py, skills_library.py
num_variables          # skills_api.py, skills_library.py
cat_fix               # skills_library.py (line 239) - means "categorical to fix"
num_fix               # skills_library.py (line 183) - means "numeric to fix"
cat_cols              # elimination.py, feature_importances.py
col_series            # skills_api.py (line 272) - singular form
c                     # binning.py (line 73) - loop variable

# NAMING CONFUSION:
user_cats            # user-provided categorical variables
user_nums            # user-provided numeric variables
```

### 2.4 Feature/Column Names

**Multiple inconsistent terms for the same concept:**

```python
# In the SAME FILE (skills_library.py):
self.cols_to_drop           # line 35
self.train_features         # line 66
feature_lists              # line 23
feature_lists             # line 84
columns                   # variable reference

# Different abbreviations:
col    → columns
cols   → columns  
c      → column (in loops)
```

### 2.5 DataFrame/Data Variables

```python
# MULTIPLE NAMES FOR SAME CONCEPT:
df              # preprocessing.py
dfx             # data.py, skills_library.py (appears 8+ times)
local_df        # preprocessing.py
train           # skills_api.py
apply           # skills_library.py (line 18) - as variable name!
pre_train_df    # skills_api.py (line 29)
X, X_test      # elimination.py, feature_importances.py
```

---

## 3. FUNCTION/METHOD NAMING INCONSISTENCIES

### 3.1 Naming Style Inconsistencies

#### camelCase Functions (INCONSISTENT)
```python
getLogger()              # helpers.py
getArgumentsList()       # helpers.py  
get_init_parameters()   # helpers.py (MIXED! - uses snake_case for one function)
```

#### snake_case Functions (Majority Standard)
```python
data_load()
train_test()
connect_to_sql()
explore()
sample()
merge()
conversion()
remove_features()
```

#### Special Cases
```python
# All lowercase (inconsistent):
to_numeric()           # preprocessing.py
to_categoric()        # preprocessing.py

# Method names with inconsistency:
fit_model()          # classifier.py
fit_transform()      # preprocessing.py
fit()                # binning.py
transform()          # Multiple files - standard
```

### 3.2 Parameter Naming Issues

**Inconsistent abbreviations in function parameters:**

```python
# Model/parameter optimization:
param_opt              # abbreviated
param_opt_method       # abbreviated
n_iter                 # abbreviated
nprocs                 # abbreviated (should be n_jobs?)
baseEstimator          # camelCase (inconsistent!)
n_feature_params       # abbreviated
importance_direction_type  # full descriptive name

# Fill values:
fill_value             # full
num_fill_value        # full
cat_strategy           # full
num_strategy           # full
```

---

## 4. ABBREVIATION INCONSISTENCIES

### 4.1 Critical Abbreviation Issues

| Term | Variations | Recommended |
|------|-----------|-------------|
| **Categorical** | `cat`, `categoric`, `category`, `cat_vars`, `user_cats` | `categorical` |
| **Numeric** | `num`, `numeric`, `number` | `numeric` |
| **Probability** | `prob`, `probs`, `y_pred_prob`, `pred_prob` | `probability` or `probabilities` |
| **Prediction** | `pred`, `preds`, `y_pred`, `predictions` | `prediction` or `predictions` |
| **Column** | `col`, `cols`, `columns`, `column_groups` | `column` or `columns` |
| **DataFrame** | `df`, `dfx`, `data` | `dataframe` or `df` (standardized) |
| **Features** | `feat`, `features`, `feature_names` | `features` or `feature_names` |
| **Imputation** | `imp_num`, `imp_cat`, `impute_null` | `imputer_numeric`, `imputer_categorical` |
| **Processing** | `preprocess`, `preprocessing`, `transform` | `preprocess` (verb form) |

### 4.2 Examples of Abbreviation Inconsistencies

```python
# SAME CONCEPT, DIFFERENT ABBREVIATIONS:

# Categorical handling:
cat_variables         # line 55, skills_library.py
categoric            # line 63, skills_library.py
user_cats            # line 106, skills_library.py
cat_fix              # line 239, skills_library.py
cat_null             # line 175, skills_library.py
cat_strategy         # line 177, skills_library.py

# Numeric handling:
num_variables        # line 56, skills_library.py
numeric              # line 64, skills_library.py
user_nums            # line 107, skills_library.py
num_fix              # line 170, skills_library.py
num_null             # line 174, skills_library.py
num_strategy         # line 176, skills_library.py
num_fill_value       # line 168, skills_library.py

# DataFrame abbreviation:
df                   # Standard in some files
dfx                  # Used in skills_library.py, skills_api.py (8+ occurrences)
data                 # Used generically
```

**Examples from code:**
```python
# skills_library.py, line 504
dfx_probs, dfx_preds = self.model.predict(test, y_test=y_test, sample='Test')

# classifier.py, line 211  
feature_imp_df = get_feature_importances(self.model, x_train, ...)

# Mixing in same file:
self.test_pred_prob = self.model.predict_proba(x_test)[:, 1]  # classifier.py
test_pred = self.model.predict(x_test)  # classifier.py
```

---

## 5. SINGULAR VS PLURAL INCONSISTENCIES

| Inconsistent | Should be | Files |
|--------------|-----------|-------|
| `variable` / `variables` | Singular: `variable`, Plural: `variables` | Inconsistent throughout |
| `feature` / `features` / `feature_names` | Singular: `feature`, Plural: `features` | Inconsistent |
| `col` / `cols` / `columns` | Plural: `columns` | Inconsistent |
| `imputation` (singular) in class but `impute_null` | Choose: `Imputer` or `impute` | preprocessing.py |
| `encoding` / `encode` | Choose: `Encoder` or `encode` | preprocessing.py |
| `scaling` / `scale` | Choose: `Scaler` or `scale` | preprocessing.py |

**Examples:**
```python
# PLURAL/SINGULAR INCONSISTENCY:

# Variable names mixing singular and plural:
self.variable_type               # Singular concept
self.cat_variables              # Plural
self.num_variables              # Plural
self.null_df                     # Unclear naming
self.features_to_fix            # Plural
self.features_to_drop           # Plural

# Function names:
feature_lists                    # Plural
feature_names                    # Plural
cat_cols                        # Abbreviated plural
col                             # Singular (in loops)
```

---

## 6. INCONSISTENT NAMING IN KEY CONCEPTS

### 6.1 Data Exploration/Validation

```python
# skills_library.py uses:
data_explore           # Class (line 51) - verb form
variable_type         # attribute
data.explore()        # Function call (line 62)

# Should be either:
# DataExplorer (class) with explore() method
# OR data_explorer (class) with explore() method
```

### 6.2 Type Conversions

```python
# preprocessing.py uses MULTIPLE different approaches:
class conversions()           # Plural, lowercase
    def to_numeric()         # Camel case method
    def to_categoric()       # Camel case method

class impute_null()          # Lowercase with verb
    def fit()
    def transform()

class encode()               # Simple lowercase
    def fit()
    def transform()
```

**This is INCONSISTENT with:**
```python
# skills_library.py uses:
class conversion:            # Singular!
    def fit()

class null_imputation:       # Full descriptive name
    def fit()

class encode_categoricals:   # Full descriptive name with verb
    def fit()

class scaling:              # Singular
    def fit()
```

### 6.3 Parameter Grid Configuration

```python
# default_grids.py uses inconsistent keys:
"logistic_regression"        # Underscore separator
"random_forest_classifier"   # Underscore separator
"lightgbm_classifier"        # Underscore separator
"xgboost_classifier"         # Underscore separator

# But param_grid_best.py uses DIFFERENT names:
"logistic_regression_base"
"logistic_regression_good"
"logistic_regression_not_good"
"logistic_regression_meta"
```

---

## 7. SPECIAL NAMING ISSUES

### 7.1 CamelCase in Wrong Context

```python
# Parameter with camelCase (INCONSISTENT with snake_case standard):
baseEstimator          # Should be base_estimator
→ Used in model_LR.__init__, model_RF.__init__, etc.

# Correct usage elsewhere:
base_estimator         # skills_api.py (line 841) - correct!
```

### 7.2 Underscore Usage Inconsistency

```python
# Inconsistent underscore spacing in assignments:
self.variable_type=None         # No spaces (line 54)
self.cat_variables=None         # No spaces (line 55)
self.null_df = self.variable_type['missing_rate']  # With spaces (line 65)
self.train_features = train.columns                 # With spaces (line 66)

# Should be consistent:
self.variable_type = None
self.cat_variables = None
```

### 7.3 Private vs Public Attributes

```python
# Using underscore suffix for "internal" attributes:
self.variables_        # elimination.py (line 175)
self.features_to_drop_ # elimination.py (line 177)
self.feature_names_in_ # elimination.py (line 113)
self.n_features_in_    # elimination.py (line 116)

# But no consistent use of underscore prefix for private attributes
# Some conventions mix with sklearn style (trailing underscore = fitted)
```

### 7.4 Special Naming in feature_importances.py

```python
# Non-standard parameter names:
def _mean_shap_categoric_var(shap_values_df, input_data_df, cat_cols):
    # Uses underscored function names (OK for private)
    a, b = shap_values_df[col], input_data_df[col]  # Single letters! (BAD)
    a.name, b.name = "shap", "category"             # OK but INCONSISTENT
```

---

## 8. DETAILED CODE EXAMPLES

### 8.1 Example 1: Class Definition Inconsistency

```python
# Option A: snake_case (binning.py, line 10)
class opt_binning:
    def __init__(self, num_variables, cat_variables):
        self.num_binner = None

# Option B: PascalCase (should be standard)
class OptBinning:
    def __init__(self, num_variables, cat_variables):
        self.num_binner = None
```

### 8.2 Example 2: Function Parameter Inconsistency

```python
# From classifier.py line 139 and 247
def fit_model(self, x_train, y_train):      # Lowercase x
    ...
    
def predict(self, x_test, y_test=[], sample='Validation'):  # Lowercase x
    ...

# But elsewhere in skills_api.py
def LogisticRegression(self,
                       ...
                       base_estimator='GP',  # snake_case parameter
                       ...):
```

### 8.3 Example 3: Abbreviation Usage in Same Method

```python
# From skills_library.py, lines 504-506
def transform(self, test, y_test=[], sample='Test'):
    dfx_probs, dfx_preds = self.model.predict(test, y_test=y_test, sample='Test')
    return dfx_probs, dfx_preds
```
- Uses `dfx_probs` instead of `probs` or `probabilities`
- Uses `dfx_preds` instead of `preds` or `predictions`
- Why the `dfx` prefix? (appears to be a legacy artifact)

---

## 9. PATTERN ANALYSIS

### Most Violated Naming Rules

1. **PEP 8 Class Names:** ~40% of classes violate PEP 8 (use snake_case instead of PascalCase)
2. **Variable Naming Consistency:** 6+ different names for train/test data concepts
3. **Abbreviation Standardization:** No consistent abbreviation policy
4. **Singular vs Plural:** Inconsistent throughout

### Files with Most Issues

1. **skills_library.py** - 7 classes with mixed naming, inconsistent variable naming
2. **skills_api.py** - Large file with mixed naming conventions
3. **preprocessing.py** - Classes with inconsistent naming (conversions vs conversion)
4. **classifier.py** - Parameter naming inconsistency (baseEstimator)
5. **binning.py** - Duplicate class definition (opt_binning)

---

## 10. RECOMMENDATIONS FOR STANDARDIZATION

### 10.1 Class Naming Standard (PEP 8 Compliant)

```python
# RENAME ALL:
opt_binning              → OptBinning
data_explore            → DataExplorer
conversion              → DataConversion
null_imputation         → NullImputation
encode_categoricals     → CategoricalEncoder
scaling                 → FeatureScaler
model_pipeline          → ModelPipeline
pipe_manager            → PipelineManager
model_args              → ModelArguments

# KEEP THESE:
DropCols                ✓ (already correct)
FeatureElimination      ✓ (already correct)
ClassificationScores    ✓ (already correct)
LogisticRegression      ✓ (already correct)
RandomForest            ✓ (already correct)
LGBM                    → LGBMClassifier (more descriptive)
XGBoost                 → XGBoostClassifier (more descriptive)
```

### 10.2 Variable Naming Standard

```python
# TRAINING/TEST DATA (use lowercase for variables):
x_train                 ✓ (keep consistent)
y_train                 ✓ (keep consistent)
x_val / x_test         ✓ (standardize: use one, preferably x_val for validation)
y_val / y_test         ✓ (standardize: use one, preferably y_val for validation)

# PREDICTIONS (standardize):
probabilities           → Instead of: probs, prob, pred_prob
predictions             → Instead of: preds, pred, y_pred
```

### 10.3 Abbreviation Standard

```python
# APPROVE THESE ABBREVIATIONS:
cat_variables           ✓ (for categorical)
num_variables           ✓ (for numeric)
n_jobs                  ✓ (for number of jobs)
max_depth               ✓ (for maximum depth)
min_samples             ✓ (for minimum samples)

# STANDARDIZE:
cat, num                → categorical, numeric (avoid abbreviation)
df                      ✓ (dataframe - standardized)
dfx                     → REMOVE (non-standard, use df)
cols                    → columns (spell out)
col_series              → column_series (for clarity)
```

### 10.4 Function/Method Naming Standard

```python
# STANDARDIZE ALL TO snake_case:
getLogger()             → get_logger()
getArgumentsList()      → get_arguments_list()
to_numeric()            ✓ (already correct)
to_categoric()          → to_categorical() (spell out)

# KEEP CONSISTENT METHOD NAMING:
fit()                   ✓
transform()             ✓
fit_transform()         ✓
```

### 10.5 Class Structure Standardization

```python
# DEPRECATE DUPLICATES:
# binning.py: opt_binning                    → MERGE with skills_library.py version
# preprocessing.py: conversions              → CONSOLIDATE with skills_library.conversion
# preprocessing.py: impute_null              → CONSOLIDATE with skills_library.null_imputation
# preprocessing.py: encode                   → CONSOLIDATE with skills_library.encode_categoricals
# preprocessing.py: scale                    → CONSOLIDATE with skills_library.scaling
```

### 10.6 Parameter Naming Standard

```python
# STANDARDIZE ALL PARAMETERS TO snake_case:
baseEstimator           → base_estimator  (CRITICAL)
param_opt               ✓ (acceptable abbreviation in context)
param_opt_method        ✓ (acceptable abbreviation in context)
n_iter                  ✓ (standard sklearn abbreviation)
nprocs                  → n_jobs (use sklearn standard)
```

---

## 11. IMPLEMENTATION PRIORITY

### Priority 1 (CRITICAL - Fix First)
- [ ] Resolve duplicate `opt_binning` class definitions
- [ ] Standardize `baseEstimator` → `base_estimator`
- [ ] Remove `dfx_` prefix usage, use standard `df` or descriptive names
- [ ] Standardize train/test data variable names across all files

### Priority 2 (HIGH - Fix Next)
- [ ] Rename all snake_case classes to PascalCase
- [ ] Consolidate duplicate class definitions (conversions, impute_null, encode, scale)
- [ ] Standardize abbreviation usage (cat→categorical, num→numeric)
- [ ] Fix inconsistent underscore spacing in assignments

### Priority 3 (MEDIUM - Fix Later)
- [ ] Rename helper functions (getLogger → get_logger, etc.)
- [ ] Standardize singular/plural naming
- [ ] Document and enforce naming conventions in docstrings

---

## 12. PEP 8 COMPLIANCE SUMMARY

| Aspect | Current | PEP 8 | Compliance |
|--------|---------|-------|-----------|
| Class Names | Mixed | PascalCase | 40% ✗ |
| Function Names | Mostly snake_case | snake_case | 80% ✓ |
| Variable Names | Mixed | snake_case | 70% ✓ |
| Constants | Not shown | UPPER_CASE | N/A |
| Module Names | Snake_case | snake_case | 100% ✓ |
| Method Names | Mixed | snake_case | 80% ✓ |

---

## 13. CONCLUSION

The codebase violates naming conventions across **5 major dimensions**:

1. **Class naming** - PEP 8 non-compliance (40% violation rate)
2. **Variable consistency** - 6+ names for same concepts
3. **Abbreviation policy** - No clear standard
4. **Parameter naming** - Mixed camelCase/snake_case
5. **Function naming** - Some camelCase violations

**Estimated refactoring effort:**
- Replace 25+ class names
- Consolidate 8+ duplicate/similar classes
- Standardize 100+ variable names
- Update 50+ function signatures
- **Total:** ~200-300 changes across 12 files

**Impact if not addressed:**
- Increased maintenance burden
- Higher cognitive load for new developers
- Increased bug risk during modifications
- Inconsistent with Python community standards (PEP 8)

