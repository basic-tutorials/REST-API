# ERROR HANDLING AND LOGGING INCONSISTENCIES ANALYSIS

## Executive Summary
The codebase shows inconsistent error handling patterns and mixed logging/print statements. While some files implement proper logging, others use print statements directly. Bare except clauses and silent error handling present security and debugging risks.

---

## CRITICAL ISSUES

### 1. BARE EXCEPT CLAUSES
**Severity: HIGH** - These catch all exceptions including KeyboardInterrupt and SystemExit

#### data.py:13-16
```python
try:
    cx_Oracle.init_oracle_client(lib_dir=os.getenv("cx_oracle_client_path"))
except:
    pass
```
**Issue**: Bare except that silently swallows all errors
**Impact**: Silent failures during Oracle client initialization
**Recommendation**: Catch specific exceptions (Exception) with proper logging

#### preprocessing.py:28
```python
def to_numeric(self, df, cols):
    try:
        df[cols].fillna(-987654321,inplace=True)
        df[cols] = df[cols].astype(int)
        df[cols].replace(-987654321,np.nan)
    except:  # Line 28
        try:
            df[cols].fillna(-987654321,inplace=True)
            df[cols] = df[cols].astype(float)
            df[cols].replace(-987654321,np.nan)
        except:  # Line 33
            logger.info("Couldn't convert to numeric.")
```
**Issue**: Nested bare except clauses without proper error logging
**Impact**: Silent conversion failures; difficult to debug data type issues
**Recommendation**: Log specific exception types and details at ERROR level

#### preprocessing.py:42
```python
def to_categoric(self, df, cols):
    try:
        df[cols] = df[cols].astype(str)
    except:  # Line 42
        logger.info("Couldn't convert to categoric.")
```
**Issue**: Bare except with info-level logging of an error condition
**Impact**: Error conditions logged as INFO instead of ERROR; exceptions hidden
**Recommendation**: Use logger.error() and catch specific exceptions

#### skills_library.py:72
```python
try:
    corr = train[col].corr(y_train)
except:  # Line 72
    corr = np.nan
```
**Issue**: Silent exception handling for data exploration; errors masked
**Impact**: Correlation calculation failures go unlogged
**Recommendation**: Log warnings for skipped columns

---

## ERROR HANDLING ISSUES

### 2. SILENT ERROR SWALLOWING

#### data.py:110-112
```python
except Exception as e:
    logger.info("Check your data path\n")
    logger.info("Data Load Error! %s" % e)
    # NO RETURN or RAISE - function continues silently
```
**Issue**: Exception logged but not re-raised; function returns None
**Impact**: Caller has no indication of failure
**Severity: HIGH**
**Recommendation**: Explicitly raise or return None with clear indication

#### data.py:63-65
```python
except Exception as e:
    logger.info("Check your sql connection or data (table name)\n")
    logger.info("Data Load Error! %s" % e)
    # NO RETURN statement
```
**Issue**: Same pattern - error logged but function returns None implicitly
**Severity: HIGH**

### 3. EXCEPTION RE-RAISING ANTI-PATTERNS

#### data.py:29-34
```python
try:
    engine.execute('select 1 from dual')
    logger.info("Connected!\n")
except Exception as e:
    raise e  # Anti-pattern: unnecessary re-raise
    logger.info("Connection error! %s" % e)  # Unreachable code!
```
**Issue**: Code after raise is unreachable; improper error handling flow
**Severity: MEDIUM**
**Recommendation**: Remove unnecessary raise or fix the exception flow

---

## LOGGING INCONSISTENCIES

### 4. MIXED LOGGING vs PRINT STATEMENTS

#### kpi_calculator.py:23-31
```python
def cv_score(estimator_, x_train, y_train):
    print("Results of ", estimator_)  # Line 23
    print()  # Line 24
    print("===Cross-val Scores:=== \n")  # Line 25
    cv_scores = cross_val_score(...)
    print(cv_scores)  # Line 27
    print("Average CV Score is: ",cv_scores.mean())  # Line 28
    print()  # Line 29
    print("Average CV gini is ",(2*cv_scores.mean()-1))  # Line 30
```
**Issue**: Using print() instead of logging
**Severity: MEDIUM**
**Recommendation**: Replace all print() with logger.info()

#### data.py:81, 93, 105, 107
```python
print("Dataframe created!\n")  # Line 81
logger.info("Dataframe created!\n")  # Duplicate output
...
print(df.shape)  # Line 107 - mixed logging and printing
```
**Issue**: Both print and logger used; duplicate output to console
**Severity: MEDIUM**
**Recommendation**: Use logger exclusively

#### kpi.py:13-42
```python
def roc_auc_score(y_true, y_pred_prob):
    roc_score=roc_auc_score(y_true, y_pred_prob)
    print(print("\n Roc Auc Score: ",roc_score))  # Line 13 - NESTED PRINT!
    return roc_score

def gini_score(y_true, y_pred_prob):
    gini=2*roc_auc_score(y_true, y_pred_prob)-1
    print("\n Gini: ",gini)  # Line 18

def confusion_matrix(y_true,y_pred):
    ...
    print("/n Confusion Matrix: ",CM)  # Line 27 - escape sequence error!
    return CM

def lift_score(y_true, y_pred):
    ...
    print("\n Lift score: ",(lift_score))  # Line 37

def ks_statistic(predict_prob, y_true):
    ...
    print("\n KS Statistics: ",ks_statistic)  # Line 42
```
**Issue**: Multiple issues: nested print(), wrong escape sequences, logging level mismatch
**Severity: MEDIUM-HIGH**

#### pipelines.py:111-114
```python
print("Best pipe for "+str(sample) +' sample: '+ str(best_pipe_id))  # Line 111
print("Best model for "+str(sample) +' sample: '+ str(best_mdl_id))  # Line 112
print("Best model algorithm for "+str(sample) +' sample: '+ str(best_algo))  # Line 113
print("Best " +str(metric)+ " for " + str(sample) +' sample: ' + str(best_metric))  # Line 114
```
**Issue**: Critical results printed instead of logged
**Severity: MEDIUM**

#### feature_importances.py:133
```python
warnings.warn("Model is linear, disabling fast option...") if fast else print("")
```
**Issue**: Inconsistent use of warnings vs print
**Severity: LOW-MEDIUM**

#### preprocessing.py:116
```python
print("Encoding completed.\n")  # Line 116 - should be logger
logger.info("Encoding completed.\n")  # But logger is also called below
```
**Issue**: Duplicate output and mixed logging methods

---

## LOGGING LEVEL INCONSISTENCIES

### 5. ERROR CONDITIONS LOGGED AS INFO

#### data.py:31-34
```python
except Exception as e:
    raise e
    logger.info("Connection error! %s" % e)  # ERROR should use logger.error()
```
**Severity: MEDIUM**

#### data.py:64-65
```python
except Exception as e:
    logger.info("Check your sql connection or data (table name)\n")  # Should be ERROR
    logger.info("Data Load Error! %s" % e)  # Should be ERROR
```
**Severity: MEDIUM**

#### preprocessing.py:34
```python
except:
    logger.info("Couldn't convert to numeric.")  # Should be ERROR or WARNING
```
**Severity: MEDIUM**

#### preprocessing.py:43
```python
except:
    logger.info("Couldn't convert to categoric.")  # Should be WARNING or ERROR
```
**Severity: MEDIUM**

#### classifier.py:272
```python
except Exception as e:
    logger.info(e)  # Bare exception info - should include context and use ERROR
```
**Severity: MEDIUM**

#### classifier.py:244-245
```python
logger.info("With ROC: " % roc_auc_score(y_train, self.train_prob))  # Wrong format string
logger.info("With gini: " % (self.train_gini))  # Same issue - will not print values!
```
**Issue**: Format string errors - % operators have nothing on left side
**Severity: HIGH** (metrics not being logged)

---

## MISSING ERROR HANDLING IN CRITICAL SECTIONS

### 6. MISSING INPUT VALIDATION

#### data.py:18
```python
def connect_to_sql(username='username', password='password'):
    # NO VALIDATION of username/password
    # NO CHECK for None values
```
**Severity: MEDIUM**

#### data.py:114
```python
def train_test(df, test_size = None, random_state=1, split_style ='random', group = None, ...):
    if test_size<1:  # NO VALIDATION that test_size is numeric
        test_size=round(len(df)*test_size)
    # If test_size is None, this will crash
```
**Severity: MEDIUM**

#### data.py:220
```python
def merge(left, right, how="inner", on=None):
    # No validation of parameters before use
    if type(left) is pd.Series:  # Inconsistent - uses 'is' instead of isinstance
```
**Severity: LOW-MEDIUM** (type checking is imprecise)

#### preprocessing.py:20-44
```python
def to_numeric(self, df, cols):
    # No validation that cols exist in df
    # No validation that df is a DataFrame
    try:
        df[cols].fillna(-987654321,inplace=True)
```
**Severity: MEDIUM**

#### skills_api.py:260-262
```python
def data_read_sql(self, username=os.getenv('EDW_USER'), ...):
    logger = getLogger()
    if not query:
        logger.critical("query parameter is required")  # Uses .critical() correctly
        raise Exception("query parameter is required")  # Good pattern!
```
**Note**: This is a GOOD example - good error message and raising

---

## INCONSISTENT ERROR MESSAGE PATTERNS

### 7. VAGUE ERROR MESSAGES

#### skills_library.py:355
```python
raise Exception(f'{eliminator} is not a valid eliminator.')
# Better: Include list of valid eliminators
```

#### skills_library.py:359
```python
raise Exception(f'{key} is not a valid parameter for {eliminator}.')
# Better: Include list of valid parameters
```

#### preprocessing.py:59
```python
raise Exception("Imputation type is not valid")
# Better: Include list of valid types
```

### 8. INCONSISTENT EXCEPTION TYPES

- Some code uses `Exception` (generic)
- Some uses `ValueError` (specific)
- Some uses `Exception` for type errors (should be TypeError)
- Inconsistent approach across similar functions

---

## TYPE CHECKING INCONSISTENCIES

### 9. POOR TYPE CHECKING PATTERNS

#### data.py:222-248
```python
if type(left) is pd.Series:  # Using 'is' with type() is risky
    left = pd.DataFrame(left)

if type(right) is pd.Series:  # Better: isinstance(left, pd.Series)
    right = pd.DataFrame(right)

# Multiple type checks instead of consistent isinstance()
if type(left) is pd.DataFrame and type(right) is pd.DataFrame:
elif type(left) is pd.DataFrame and type(right) is np.ndarray:
elif type(left) is np.ndarray and type(right) is pd.DataFrame:
```
**Recommendation**: Use isinstance() consistently

#### skills_api.py:600, 636, 909, 1021, 1131, 1241
```python
if type(param_grid) == str:  # Inconsistent - uses == with type()
```
**Recommendation**: Use isinstance(param_grid, str)

#### skills_library.py:339
```python
if not isinstance(self.params["estimator"], str):  # Good - correct pattern
```
**Note**: This is correct, but inconsistent with other code in same project

---

## MISSING LOGGING IN CRITICAL PATHS

### 10. SILENT FAILURES IN DATA PROCESSING

#### preprocessing.py:12
```python
def remove_features(df, features, inplace):
    df.drop(columns=features)  # Drop not assigned back - NO ERROR
    return df  # Returns unchanged df - SILENT FAILURE
```
**Severity: HIGH** - Feature removal silently fails!

#### preprocessing.py:27, 32
```python
df[cols].replace(-987654321,np.nan)  # Result not assigned - SILENT FAILURE
# Should be: df[cols] = df[cols].replace(...)
```
**Severity: HIGH** - Replace operations don't update dataframe!

#### skills_api.py:455
```python
else:
    pass  # Silent passthrough - no logging
```
**Severity: LOW** (context dependent)

---

## LOGGING CONFIGURATION GAPS

### 11. INCONSISTENT LOGGER INSTANTIATION

#### Multiple files
```python
# Pattern 1: New instance each time
logger = getLogger()

# Pattern 2: Retrieve at module level (in helpers.py)
logger = logging.getLogger('qnbanalytics')
```
**Issue**: Multiple logger instances possible; inconsistent configuration
**Recommendation**: Use module-level logger consistently

---

## SUMMARY TABLE OF FINDINGS

| File | Issue | Line | Severity | Type |
|------|-------|------|----------|------|
| data.py | Bare except + pass | 15 | HIGH | Bare except |
| data.py | Silent data load failure | 110-112 | HIGH | Silent error |
| data.py | Format string error | 244-245 | HIGH | Format error |
| preprocessing.py | Nested bare excepts | 28, 33 | HIGH | Bare except |
| preprocessing.py | Silent replace failure | 27, 32 | HIGH | Silent failure |
| preprocessing.py | Wrong log level | 34, 43 | MEDIUM | Log level |
| preprocessing.py | Print + logger mix | 116 | MEDIUM | Mixed logging |
| kpi.py | Nested print() | 13 | MEDIUM | Print statement |
| kpi.py | Escape sequence error | 27 | MEDIUM | Syntax issue |
| kpi_calculator.py | All print statements | 23-31 | MEDIUM | Print statements |
| pipelines.py | Print results | 111-114 | MEDIUM | Print statements |
| skills_library.py | Silent exception | 72 | MEDIUM | Silent error |
| skills_library.py | Vague exception | 355, 359 | LOW | Error message |
| feature_importances.py | Print vs warning | 133 | LOW-MEDIUM | Mixed logging |
| skills_api.py | Type checking | 600, 636 | LOW | Type checking |
| data.py | Type checking | 222-248 | LOW | Type checking |

---

## RECOMMENDATIONS

1. **IMMEDIATE (Critical)**
   - Fix format string errors in classifier.py:244-245
   - Fix silent failures in preprocessing.py:27, 32, 12
   - Replace all bare except clauses with specific exceptions
   - Add proper logging to data loading failures

2. **HIGH PRIORITY**
   - Replace all print() statements with logger calls
   - Change error-level log statements from INFO to ERROR/WARNING
   - Add input validation to critical functions
   - Fix exception re-raising patterns

3. **MEDIUM PRIORITY**
   - Standardize exception types (use ValueError, TypeError, etc.)
   - Improve error messages with context information
   - Make logger instantiation consistent
   - Add context managers for resource management

4. **BEST PRACTICES**
   - Use isinstance() instead of type() for type checking
   - Include exc_info=True in logger calls for tracebacks
   - Avoid silent error handling (always log and/or raise)
   - Document expected exceptions in docstrings

