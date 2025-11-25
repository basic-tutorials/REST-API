# 📊 Categorical Encoding Setup - Using category_encoders

## ✅ **Current Status**

Your codebase **already uses `category_encoders.TargetEncoder`**!

### Implementation Details

**File**: `QNBAnalytics_ML/preprocessing.py`
**Method**: `TargetEncoder` from category_encoders library
**Configuration**:
```python
from category_encoders import TargetEncoder

class encode():
    def __init__(self, method="Target Encoder", sample=50000, min_samples_leaf=50):
        self.encoder = TargetEncoder(min_samples_leaf=min_samples_leaf)
```

### How It Works

1. **Automatic Detection**: Pipeline detects object/categorical dtypes
2. **Target Encoding**: Maps each category to mean target value
3. **Regularization**: `min_samples_leaf=50` prevents overfitting on rare categories
4. **Sampling**: Uses 50K samples for fitting (faster on large datasets)
5. **Handles Unseen**: Automatically handles new categories in test set

---

## 🎯 **New Categorical Features to Encode**

Your data has 4 new categorical features that will benefit from TargetEncoder:

| Feature | Type | Unique Values | Encoding Strategy |
|---------|------|---------------|-------------------|
| **WORKGROUP** | Categorical (String) | ~2-3 | ✅ Target encode - employment stability signal |
| **QRUP** | Categorical (String) | ~5-10 | ✅ Target encode - organization type signal |
| **SAHƏLƏR** | Categorical (String) | ~20-30 | ✅ Target encode - industry risk profile |
| **SEKTOR** | Categorical (String) | ~30-50 | ✅ Target encode - detailed sector risk |

### Why TargetEncoder is Perfect Here

**Advantages**:
- ✅ Handles high cardinality (SAHƏLƏR, SEKTOR have many unique values)
- ✅ Preserves predictive information (unlike one-hot encoding)
- ✅ Works with tree models AND linear models
- ✅ Regularization prevents overfitting on rare categories
- ✅ Automatically handles missing values

**Example**: SAHƏLƏR encoding
```
"Neft-qaz sektoru" (Oil/Gas) → 0.015 (1.5% default rate)
"Dövlət qurumları" (State org) → 0.008 (0.8% default rate)
"Digər" (Other) → 0.025 (2.5% default rate)
```

---

## 🔧 **Configuration in Training Notebook**

### Step 1: Enable Categorical Encoding (Cell ~10)

**Find this cell** (ID: `dc648bc2-d752-40f9-a873-3da198406c48`):

```python
## PIPELINE SETTINGS

apply_categorical_encoding = False  # ❌ CURRENTLY DISABLED
```

**Change to**:
```python
## PIPELINE SETTINGS FOR LAYER 1 (BASE MODEL)

apply_data_explore = True
apply_null_elimination = True
apply_constant_elimination = True
apply_low_gini_elimination = True
apply_correlated_feature_elimination = True
apply_binning = True
apply_scaling = False
apply_null_imputation = True        # ✅ Handle missing MAAS values
apply_categorical_encoding = True   # ✅ ENABLE for WORKGROUP, QRUP, SAHƏLƏR, SEKTOR

apply_LR = True
```

### Step 2: Ensure Correct Processing Order (Cell ~11)

**CRITICAL**: Categorical encoding must happen **BEFORE** WOE binning!

**Find in `Pipeline_Base.train()` method**:
```python
########################   Feature Elimination    ############################

if apply_correlated_feature_elimination:
    self.pipeline.feature_elimination(...)

########################   Binning    ############################  ⚠️ TOO EARLY!

if apply_binning:
    self.pipeline.binning()

########################   Categorical Encoding    ############################  ❌ TOO LATE!

if apply_categorical_encoding:
    self.pipeline.encode_categoricals()
```

**Change to this order**:
```python
########################   Feature Elimination    ############################

if apply_correlated_feature_elimination:
    self.pipeline.feature_elimination(...)


########################   Null Imputation    ############################
# ✅ STEP 1: Handle missing values first

if apply_null_imputation:
    self.pipeline.null_imputation()


########################   Categorical Encoding    ############################
# ✅ STEP 2: Encode categorical features (WORKGROUP, QRUP, SAHƏLƏR, SEKTOR)

if apply_categorical_encoding:
    self.pipeline.encode_categoricals()


########################   Binning    ############################
# ✅ STEP 3: WOE binning (now works on numeric features only)

if apply_binning:
    self.pipeline.binning()  # Remove use_existing parameter!


########################   Scaling    ############################

if apply_scaling:
    self.pipeline.scaling()
```

**Why this order?**
1. **Impute first**: Fill missing MAAS values
2. **Encode second**: Convert WORKGROUP/QRUP/SAHƏLƏR/SEKTOR to numeric using TargetEncoder
3. **Bin third**: Apply WOE binning to all numeric features (including encoded ones)

---

## 📊 **How the Pipeline Detects Categorical Features**

The pipeline **automatically detects** categorical features using:

1. **Data type detection**: `object`, `string`, or `category` dtype
2. **Cardinality check**: Features with < 100 unique values
3. **Value type**: Non-numeric values

**Your new features will be auto-detected**:
```python
WORKGROUP: dtype=object, unique=3 → ✅ Categorical
QRUP: dtype=object, unique=8 → ✅ Categorical
SAHƏLƏR: dtype=object, unique=25 → ✅ Categorical
SEKTOR: dtype=object, unique=45 → ✅ Categorical
MAAS: dtype=float64 → ❌ Not categorical (numeric)
```

---

## 🔍 **Layer 2 Models (Good/Not-Good Segments)**

**Important**: Layer 2 models also have categorical encoding!

**Find in Cell ~18** (Pipeline settings for Layer 2):
```python
apply_categorical_encoding = True  # ✅ ALREADY ENABLED for Layer 2
```

**This is correct!** Layer 2 models use:
- StandardScaler (for numeric features)
- Null Imputation (for missing values)
- **TargetEncoder** (for categorical features)
- Then train LGBM/LogisticRegression

**Processing order in Layer 2**:
```
Drop features → Scale → Impute → Encode → Train model
```

---

## ⚙️ **Advanced: Tuning TargetEncoder Parameters**

If you want to customize encoding behavior, you can modify `preprocessing.py`:

### Current Settings (DEFAULT)
```python
encode(
    method="Target Encoder",
    sample=50000,              # Use 50K samples for fitting
    min_samples_leaf=50        # Regularization: require 50 samples per category
)
```

### For Your Data (RECOMMENDED)
```python
# In skills_api.py, when calling encode_categoricals():
encode(
    method="Target Encoder",
    sample=50000,              # Good - your data is 87K rows
    min_samples_leaf=30        # ⚠️ Lower if categories are rare
)
```

**When to adjust `min_samples_leaf`**:
- **Too high (>50)**: Rare categories get averaged with overall mean (loses signal)
- **Too low (<20)**: Risk of overfitting on rare categories
- **Sweet spot (30-50)**: Good balance for your 87K rows

---

## 📈 **Expected Results After Encoding**

### Feature Transformation

**BEFORE encoding**:
```
WORKGROUP: ["əsas işçi heyəti", "əlavə işçi heyəti", "", "əsas işçi heyəti", ...]
```

**AFTER TargetEncoder**:
```
WORKGROUP: [0.012, 0.028, 0.020, 0.012, ...]
            ↑       ↑       ↑       ↑
         1.2%    2.8%    2.0%    1.2%  default rate
```

### Feature Importance

**Expected ranking** in `Output/FEATURE_IMPORTANCES_BASE_training.xlsx`:

| Rank | Feature | Type | Gini | Comment |
|------|---------|------|------|---------|
| 1-5 | MAAS | Numeric | 0.15-0.25 | Salary - direct income signal |
| 5-15 | SAHƏLƏR_encoded | Categorical→Numeric | 0.08-0.15 | Industry risk profile |
| 15-25 | WORKGROUP_encoded | Categorical→Numeric | 0.05-0.10 | Employment stability |
| 20-30 | QRUP_encoded | Categorical→Numeric | 0.04-0.08 | Organization type |
| 25-40 | SEKTOR_encoded | Categorical→Numeric | 0.03-0.07 | Sector detail |

---

## ✅ **Verification Steps**

### After Training, Check:

**1. Encoding Logs**
Look for these in notebook output:
```
Encoding started...
Encoding completed.
```

**2. Feature Names in Binning Table**
Open `Output/BINNING_TABLE_BASE_training.xlsx`:
- ✅ Should see: MAAS, WORKGROUP, QRUP, SAHƏLƏR, SEKTOR
- ✅ All should have WOE bins (not raw category strings)

**3. Feature Importances**
Open `Output/FEATURE_IMPORTANCES_BASE_training.xlsx`:
- ✅ MAAS should be in top 10
- ✅ WORKGROUP, SAHƏLƏR should be in top 30

**4. Model Performance**
Check Gini improvement:
- Old: 0.563 (test)
- New: 0.580-0.610 (test)
- Improvement: +0.02 to +0.05

---

## 🚨 **Common Issues & Solutions**

### Issue 1: "Encoding failed - all values are numeric"
**Cause**: Features already encoded or not detected as categorical
**Solution**: Check dtype with `df.dtypes` - should be `object` not `float64`

### Issue 2: "TargetEncoder gives NaN values"
**Cause**: Rare categories in test set not seen during training
**Solution**: TargetEncoder automatically handles this by using mean encoding

### Issue 3: "Feature importance shows WORKGROUP_woe instead of WORKGROUP"
**Cause**: Feature went through both encoding AND binning
**Solution**: This is correct! WOE binning happens after encoding.

### Issue 4: "Overfitting on categorical features"
**Cause**: `min_samples_leaf` too low
**Solution**: Increase to 50-100 in `preprocessing.py`

---

## 📝 **Summary Checklist**

Before retraining:
- [ ] Verify `category_encoders` is installed: `pip list | grep category`
- [ ] Enable `apply_categorical_encoding = True` (Cell ~10)
- [ ] Reorder pipeline: impute → encode → bin (Cell ~11)
- [ ] Remove `use_existing='binning.pkl'` from binning call
- [ ] Backup existing models: `cp Models/*.pkl Models/backup/`
- [ ] Delete old binning: `rm Models/binning.pkl`

After retraining:
- [ ] Check encoding logs in notebook output
- [ ] Verify WORKGROUP/QRUP/SAHƏLƏR/SEKTOR in binning table
- [ ] Confirm Gini improvement of +0.02 to +0.05
- [ ] Review feature importances - new features in top 30

---

## 🎯 **Ready to Retrain?**

**category_encoders is already set up and ready to use!**

Just make the 2 changes:
1. Enable `apply_categorical_encoding = True`
2. Reorder pipeline: encode BEFORE binning

Then run the notebook and watch TargetEncoder work its magic on your categorical features! 🚀
