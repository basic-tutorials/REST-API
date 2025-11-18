# ERROR HANDLING & LOGGING - QUICK REFERENCE GUIDE

## CRITICAL ISSUES (Fix Immediately)

### 1. Format String Errors in classifier.py:244-245
```python
# BROKEN:
logger.info("With ROC: " % roc_auc_score(y_train, self.train_prob))
logger.info("With gini: " % (self.train_gini))

# CORRECT:
logger.info("With ROC: %s" % roc_auc_score(y_train, self.train_prob))
logger.info("With gini: %s" % (self.train_gini))
```
**Impact**: Metrics are not being logged at all!

### 2. Silent Data Processing Failures in preprocessing.py
```python
# BROKEN (line 12):
def remove_features(df, features, inplace):
    df.drop(columns=features)  # Not assigned!
    return df  # Returns unchanged

# BROKEN (lines 27, 32):
df[cols].replace(-987654321, np.nan)  # Result not assigned!

# CORRECT:
def remove_features(df, features, inplace):
    df = df.drop(columns=features, errors='ignore')
    return df

# CORRECT:
df[cols] = df[cols].replace(-987654321, np.nan)
```
**Impact**: Feature removal and null imputation don't work!

### 3. Bare Except Clauses (Replace ALL)
**Files**: data.py:15, preprocessing.py:28, preprocessing.py:33, preprocessing.py:42, skills_library.py:72

```python
# BROKEN:
try:
    some_code()
except:
    pass

# CORRECT:
try:
    some_code()
except Exception as e:
    logger.error("Failed to initialize Oracle client: %s", e)
```

### 4. Silent Error Swallowing in data.py:63-112
```python
# BROKEN:
except Exception as e:
    logger.info("Data Load Error! %s" % e)
    # Function returns None implicitly - caller has no idea!

# CORRECT:
except Exception as e:
    logger.error("Failed to load data from path: %s", e)
    raise  # or return None with explicit check
```

---

## HIGH PRIORITY FIXES

### 5. Replace ALL print() with logger.info()

**Files with print() statements**:
- kpi_calculator.py: Lines 23, 24, 25, 27, 28, 29, 30
- kpi.py: Lines 13, 18, 27, 37, 42
- pipelines.py: Lines 111, 112, 113, 114
- data.py: Lines 81, 93, 105, 107
- preprocessing.py: Line 116
- feature_importances.py: Line 133

```python
# BROKEN:
print("Results of ", estimator_)

# CORRECT:
logger = getLogger()
logger.info("Results of %s", estimator_)
```

### 6. Fix Logging Levels

**BROKEN** - Using logger.info() for errors:
```python
except Exception as e:
    logger.info("Data Load Error! %s" % e)  # Should be ERROR!
```

**CORRECT**:
```python
except Exception as e:
    logger.error("Data Load Error: %s", e)  # Use ERROR level
```

### 7. Improve Exception Messages

**BROKEN** - Vague messages:
```python
raise Exception("Imputation type is not valid")
```

**CORRECT** - Include context:
```python
valid_types = ['simple', 'advanced']
if self.type not in valid_types:
    raise ValueError(f"Invalid imputation type '{self.type}'. Valid options: {valid_types}")
```

---

## MEDIUM PRIORITY FIXES

### 8. Fix Type Checking

**BROKEN** - Using type() with is/==:
```python
if type(left) is pd.Series:
if type(param_grid) == str:
```

**CORRECT** - Use isinstance():
```python
if isinstance(left, pd.Series):
if isinstance(param_grid, str):
```

### 9. Add Input Validation

**BROKEN** - No parameter validation:
```python
def train_test(df, test_size=None, ...):
    if test_size < 1:  # Will crash if test_size is None!
```

**CORRECT** - Validate inputs:
```python
def train_test(df, test_size=None, ...):
    if test_size is None:
        raise ValueError("test_size parameter is required")
    if not isinstance(test_size, (int, float)):
        raise TypeError(f"test_size must be int or float, got {type(test_size)}")
    if not (0 < test_size < 1 or test_size >= 1):
        raise ValueError("test_size must be > 0")
```

### 10. Fix Exception Re-raising Issues

**BROKEN** - Unreachable code:
```python
except Exception as e:
    raise e
    logger.info("Connection error! %s" % e)  # Never executes!
```

**CORRECT**:
```python
except Exception as e:
    logger.error("Connection failed: %s", e)
    raise  # Better than 'raise e'
```

---

## PATTERN REFERENCE

### Good Logging Pattern
```python
def my_function(param):
    logger = getLogger()
    
    # Input validation
    if param is None:
        raise ValueError("param is required")
    
    try:
        result = process(param)
        logger.info("Processing completed for %s", param)
        return result
    except ValueError as e:
        logger.error("Invalid value provided: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during processing: %s", e, exc_info=True)
        raise
```

### Bad Logging Pattern (Avoid)
```python
def my_function(param):
    try:
        print("Processing...")  # Bad: print instead of logging
        result = process(param)
        print(result)  # Bad: results printed
        return result
    except:  # Bad: bare except
        pass  # Bad: silent failure
```

---

## FILE-BY-FILE SUMMARY

| File | Issues | Lines | Action |
|------|--------|-------|--------|
| **data.py** | Bare except, silent errors, format string | 15, 63, 110, 244 | HIGH |
| **preprocessing.py** | Bare excepts, silent failures, wrong levels | 12, 27, 28, 32, 34, 42, 116 | HIGH |
| **kpi.py** | Nested print, escape sequences | 13, 27 | MEDIUM |
| **kpi_calculator.py** | All print statements | 23-31 | MEDIUM |
| **pipelines.py** | Results printed instead of logged | 111-114 | MEDIUM |
| **classifier.py** | Format string error | 244-245 | HIGH |
| **skills_library.py** | Vague exceptions, silent handling | 72, 355, 359 | LOW-MEDIUM |
| **skills_api.py** | Type checking inconsistency | 600, 636 | LOW |
| **feature_importances.py** | Mixed logging/warnings | 133 | LOW-MEDIUM |

---

## VALIDATION CHECKLIST

Before committing fixes, ensure:

- [ ] All `except:` are replaced with specific exceptions
- [ ] All `print()` calls replaced with `logger.info()`
- [ ] All error-level logs use `logger.error()` not `logger.info()`
- [ ] Format strings are correct (using %s or f-strings)
- [ ] All function parameters have type hints or validation
- [ ] Critical operations have try-except with logging
- [ ] No unreachable code after `raise`
- [ ] All method calls check return values before use
- [ ] Silent failures are eliminated (always log or raise)
- [ ] Exception messages include context and valid values

