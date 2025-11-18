# SPECIFIC FIX EXAMPLES FOR ERROR HANDLING ISSUES

## Critical Issue #1: Format String Errors (classifier.py:244-245)

### Current Code (BROKEN)
```python
# classifier.py, lines 244-245
self.train_prob = self.model.predict_proba(x_train)[:, 1]
self.train_gini = 2 * roc_auc_score(y_train, self.train_prob) - 1
logger.info("With ROC: " % roc_auc_score(y_train, self.train_prob))
logger.info("With gini: " % (self.train_gini))
```

### Fixed Code
```python
# classifier.py, lines 244-245
self.train_prob = self.model.predict_proba(x_train)[:, 1]
self.train_gini = 2 * roc_auc_score(y_train, self.train_prob) - 1
logger.info("With ROC: %s", roc_auc_score(y_train, self.train_prob))
logger.info("With gini: %s", self.train_gini)
```

**What changed**: Added `%s` format specifier before the `%` operator

---

## Critical Issue #2: Silent Data Processing Failures (preprocessing.py:12, 27, 32)

### Current Code (BROKEN)
```python
# preprocessing.py, line 12
def remove_features(df, features, inplace):
    df.drop(columns=features)  # This doesn't modify df!
    return df  # Returns unchanged dataframe

# preprocessing.py, lines 27, 32
df[cols].fillna(-987654321, inplace=True)
df[cols] = df[cols].astype(int)
df[cols].replace(-987654321, np.nan)  # Result not assigned!
```

### Fixed Code
```python
# preprocessing.py, line 12
def remove_features(df, features, inplace):
    if inplace:
        df.drop(columns=features, errors='ignore', inplace=True)
    else:
        df = df.drop(columns=features, errors='ignore')
    logger = getLogger()
    logger.info("Removed features: %s", features)
    return df

# preprocessing.py, lines 27, 32
df[cols].fillna(-987654321, inplace=True)
df[cols] = df[cols].astype(int)
df[cols] = df[cols].replace(-987654321, np.nan)  # Assign result back!
```

**What changed**: 
- Assignment of drop() result back to df
- Assignment of replace() result back to df[cols]
- Added logging for debugging

---

## Critical Issue #3: Bare Except in data.py:15

### Current Code (BROKEN)
```python
# data.py, lines 13-16
try:
    cx_Oracle.init_oracle_client(lib_dir=os.getenv("cx_oracle_client_path"))
except:
    pass
```

### Fixed Code
```python
# data.py, lines 13-16
try:
    cx_oracle_path = os.getenv("cx_oracle_client_path")
    if cx_oracle_path:
        cx_Oracle.init_oracle_client(lib_dir=cx_oracle_path)
        logger = getLogger()
        logger.info("Oracle client initialized successfully")
except ImportError:
    logger = getLogger()
    logger.warning("cx_Oracle not installed, Oracle connections unavailable")
except Exception as e:
    logger = getLogger()
    logger.error("Failed to initialize Oracle client: %s", e)
```

**What changed**:
- Replaced bare `except:` with specific exception types
- Added logging at appropriate levels
- Added parameter validation

---

## Critical Issue #4: Bare Except in preprocessing.py:28, 33

### Current Code (BROKEN)
```python
# preprocessing.py, lines 20-35
def to_numeric(self, df, cols):
    logger = getLogger()
    self.to_numeric = cols
    
    try:
        df[cols].fillna(-987654321, inplace=True)
        df[cols] = df[cols].astype(int)
        df[cols].replace(-987654321, np.nan)
    except:  # <-- Bare except catches everything!
        try:
            df[cols].fillna(-987654321, inplace=True)
            df[cols] = df[cols].astype(float)
            df[cols].replace(-987654321, np.nan)
        except:  # <-- Another bare except!
            logger.info("Couldn't convert to numeric.")
    return df
```

### Fixed Code
```python
# preprocessing.py, lines 20-35
def to_numeric(self, df, cols):
    logger = getLogger()
    self.to_numeric = cols
    
    try:
        df[cols].fillna(-987654321, inplace=True)
        df[cols] = df[cols].astype(int)
        df[cols] = df[cols].replace(-987654321, np.nan)
        logger.info("Successfully converted %s to integer type", cols)
    except (ValueError, TypeError) as e:
        logger.debug("Could not convert to integer, attempting float: %s", e)
        try:
            df[cols].fillna(-987654321, inplace=True)
            df[cols] = df[cols].astype(float)
            df[cols] = df[cols].replace(-987654321, np.nan)
            logger.info("Successfully converted %s to float type", cols)
        except (ValueError, TypeError) as e:
            logger.error("Could not convert %s to numeric type: %s", cols, e)
            # Return unchanged df and let caller decide
    return df
```

**What changed**:
- Replaced bare `except:` with specific exception types `(ValueError, TypeError)`
- Changed logging level from INFO to ERROR for actual errors
- Fixed silent assignment issues
- Added debug logging for fallback attempts

---

## Critical Issue #5: Silent Error Swallowing in data.py:63-65, 110-112

### Current Code (BROKEN)
```python
# data.py, lines 63-65
except Exception as e:
    logger.info("Check your sql connection or data (table name)\n")
    logger.info("Data Load Error! %s" % e)
    # Function returns None implicitly!

# data.py, lines 110-112
except Exception as e:
    logger.info("Check your data path\n")
    logger.info("Data Load Error! %s" % e)
    # Function returns None implicitly!
```

### Fixed Code
```python
# data.py, lines 63-65
except Exception as e:
    logger.error(
        "Failed to load data from SQL. Check connection details and table name. Error: %s", 
        e, 
        exc_info=True
    )
    raise  # Explicitly raise so caller knows about the failure

# data.py, lines 110-112
except Exception as e:
    logger.error(
        "Failed to load data from file path: %s. Error: %s", 
        data, 
        e,
        exc_info=True
    )
    raise  # Explicitly raise so caller knows about the failure
```

**What changed**:
- Changed logging level from INFO to ERROR
- Added `exc_info=True` to include full traceback
- Explicitly `raise` to propagate error to caller
- Improved error message with context

---

## High Priority Issue #1: Replace print() with logger (kpi_calculator.py:23-31)

### Current Code (BROKEN)
```python
# kpi_calculator.py, lines 21-31
def cv_score(estimator_, x_train, y_train):
    # Calculate ginis for each selected variable
    print("Results of ", estimator_)
    print()
    print("===Cross-val Scores:=== \n")
    cv_scores = cross_val_score(estimator_, x_train, y_train, cv=5, scoring='roc_auc')
    print(cv_scores)
    print("Average CV Score is: ", cv_scores.mean())
    print()
    print("Average CV gini is ", (2*cv_scores.mean()-1))
    print()
    return cv_scores
```

### Fixed Code
```python
# kpi_calculator.py, lines 21-31
def cv_score(estimator_, x_train, y_train):
    from QNBAnalytics_ML.helpers import getLogger
    logger = getLogger()
    
    # Calculate ginis for each selected variable
    logger.info("Results of: %s", estimator_)
    logger.info("Running Cross-validation Scores...")
    
    cv_scores = cross_val_score(estimator_, x_train, y_train, cv=5, scoring='roc_auc')
    logger.info("CV Scores: %s", cv_scores)
    logger.info("Average CV Score: %f", cv_scores.mean())
    
    gini = 2 * cv_scores.mean() - 1
    logger.info("Average CV Gini: %f", gini)
    
    return cv_scores
```

**What changed**:
- Replaced all `print()` with `logger.info()`
- Improved readability of log messages
- Used proper formatting for numeric values

---

## High Priority Issue #2: Fix kpi.py Logging Issues (lines 13-42)

### Current Code (BROKEN)
```python
# kpi.py, lines 11-42
def roc_auc_score(y_true, y_pred_prob):
    roc_score = roc_auc_score(y_true, y_pred_prob)
    print(print("\n Roc Auc Score: ", roc_score))  # Nested print!
    return roc_score

def gini_score(y_true, y_pred_prob):
    gini = 2*roc_auc_score(y_true, y_pred_prob)-1
    print("\n Gini: ", gini)
    return gini

def confusion_matrix(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print("/n Confusion Matrix: ", CM)  # Wrong escape sequence!
    return CM

def lift_score(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    lift_score = (TP/(TP+FP))/((TP+FN)/(TP+TN+FP+FN))
    print("\n Lift score: ", (lift_score))
    return lift_score

def ks_statistic(predict_prob, y_true):
    ks_statistic = ks_2samp(...)
    print("\n KS Statistics: ", ks_statistic)
    return ks_statistic.statistic
```

### Fixed Code
```python
# kpi.py, lines 11-42
from QNBAnalytics_ML.helpers import getLogger

class ClassificationScores():
    
    @staticmethod
    def roc_auc_score(y_true, y_pred_prob):
        logger = getLogger()
        roc_score = roc_auc_score(y_true, y_pred_prob)
        logger.info("ROC AUC Score: %f", roc_score)
        return roc_score
    
    @staticmethod
    def gini_score(y_true, y_pred_prob):
        logger = getLogger()
        gini = 2 * roc_auc_score(y_true, y_pred_prob) - 1
        logger.info("Gini Score: %f", gini)
        return gini
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        logger = getLogger()
        CM = confusion_matrix(y_true, y_pred)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        logger.info("Confusion Matrix:\nTN=%d, FP=%d\nFN=%d, TP=%d", TN, FP, FN, TP)
        return CM
    
    @staticmethod
    def lift_score(y_true, y_pred):
        logger = getLogger()
        CM = confusion_matrix(y_true, y_pred)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        lift_score = (TP/(TP+FP))/((TP+FN)/(TP+TN+FP+FN))
        logger.info("Lift Score: %f", lift_score)
        return lift_score
    
    @staticmethod
    def ks_statistic(predict_prob, y_true):
        logger = getLogger()
        ks_result = ks_2samp(...)
        logger.info("KS Statistic: %f", ks_result.statistic)
        return ks_result.statistic
```

**What changed**:
- Replaced nested `print(print(...))` with single `logger.info()`
- Fixed escape sequence from `/n` to proper newlines in message
- Removed print statements and used proper logger calls
- Used format specifiers for numeric values

---

## Medium Priority Issue: Fix Type Checking (data.py:222-248)

### Current Code (INCORRECT)
```python
# data.py, lines 222-248
if type(left) is pd.Series:  # Using 'is' with type()
    left = pd.DataFrame(left)

if type(right) is pd.Series:
    right = pd.DataFrame(right)

if type(left) is pd.DataFrame and type(right) is pd.DataFrame:
    # merge code
elif type(left) is pd.DataFrame and type(right) is np.ndarray:
    # merge code
elif type(left) is np.ndarray and type(right) is pd.DataFrame:
    # merge code
```

### Fixed Code
```python
# data.py, lines 222-248
if isinstance(left, pd.Series):  # Use isinstance()
    left = pd.DataFrame(left)

if isinstance(right, pd.Series):
    right = pd.DataFrame(right)

if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
    # merge code
elif isinstance(left, pd.DataFrame) and isinstance(right, np.ndarray):
    # merge code
elif isinstance(left, np.ndarray) and isinstance(right, pd.DataFrame):
    # merge code
```

**What changed**:
- Replaced `type(x) is SomeType` with `isinstance(x, SomeType)`
- This is more Pythonic and handles subclasses correctly

---

## Medium Priority Issue: Add Input Validation (data.py:114)

### Current Code (NO VALIDATION)
```python
# data.py, lines 114-119
def train_test(df, test_size=None, random_state=1, split_style='random', group=None, 
               column_name=None, sample=None, stratify=None):
    logger = getLogger()    
    logger.info("Train/Test Split Started")
    if test_size < 1:  # Will crash if test_size is None!
        test_size = round(len(df)*test_size)
```

### Fixed Code
```python
# data.py, lines 114-130
def train_test(df, test_size=None, random_state=1, split_style='random', group=None,
               column_name=None, sample=None, stratify=None):
    logger = getLogger()
    
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a DataFrame, got {type(df)}")
    
    if test_size is None:
        raise ValueError("test_size parameter is required")
    
    if not isinstance(test_size, (int, float)):
        raise TypeError(f"test_size must be numeric, got {type(test_size)}")
    
    if split_style not in ['random', 'group']:
        raise ValueError(f"split_style must be 'random' or 'group', got '{split_style}'")
    
    logger.info("Train/Test Split Started")
    
    if 0 < test_size < 1:
        test_size = round(len(df) * test_size)
    elif test_size < 1 or test_size >= len(df):
        raise ValueError(f"test_size invalid: {test_size}")
```

**What changed**:
- Added validation for DataFrame input
- Added checks for None values
- Added type checking with clear error messages
- Validated parameter values

---

## Testing Checklist

After making these fixes, test:

1. **data.py fixes**:
   ```python
   # Test bare except removal
   df = data_load(engine=None, data="test.csv")
   assert df is not None
   
   # Test silent error now raises
   try:
       df = data_load(engine=None, data="nonexistent.csv")
   except Exception as e:
       print(f"Error caught: {e}")
   ```

2. **preprocessing.py fixes**:
   ```python
   # Test feature removal
   result = remove_features(df, features=['col1'], inplace=False)
   assert 'col1' not in result.columns
   
   # Test conversion
   conv = conversions()
   result = conv.to_numeric(df, ['col1'])
   assert result['col1'].dtype in [int, float]
   ```

3. **Logging fixes**:
   ```python
   # Check that no print statements appear
   import subprocess
   result = subprocess.run(['grep', '-r', 'print(', 'QNBAnalytics_ML/'], 
                          capture_output=True)
   assert result.returncode == 1  # No matches found
   ```

