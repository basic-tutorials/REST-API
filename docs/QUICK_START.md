# 🚀 Quick Start - Retrain Model with New Features

## ✅ **Everything is Ready!**

- ✅ Custom WOE binning installed (no optbinning needed)
- ✅ category_encoders (TargetEncoder) already integrated
- ✅ 8 new features identified (4 to use, 4 to drop)
- ✅ Documentation created (RETRAINING_GUIDE.md, CATEGORICAL_ENCODING_GUIDE.md)

---

## 🎯 **Your New Features**

### Will ADD Predictive Power:
1. **MAAS** (numeric) - Salary → Expected TOP 5 feature
2. **WORKGROUP** (categorical) - Employment type → TargetEncoder
3. **QRUP** (categorical) - Organization type → TargetEncoder
4. **SAHƏLƏR** (categorical) - Business sector → TargetEncoder
5. **SEKTOR** (categorical) - Specific sector → TargetEncoder (maybe redundant)

### Will DROP (not predictive):
- ID, FIN, MKR_ID, MKR_DATE, IDS, SCR_CONSTANT, SCR_MULTIPLIER

---

## 📝 **2 Simple Changes to Make**

### Open `BOB_Scorecard_Training.ipynb` and:

### **Change 1**: Update `cols_to_drop` (Cell 2)
Add these ID columns to the drop list:
```python
cols_to_drop = [
    # ...existing 28 features...
    "BGN",

    # ADD THESE 7 NEW ONES:
    "ID", "FIN", "MKR_ID", "MKR_DATE", "IDS",
    "SCR_CONSTANT", "SCR_MULTIPLIER",
]
```

### **Change 2**: Fix Pipeline Order (Cell ~11)

In `Pipeline_Base.train()` method, **reorder these 3 sections**:

**WRONG ORDER (current)**:
```python
if apply_binning:
    self.pipeline.binning(use_existing='binning.pkl')  # ❌ Too early!

if apply_null_imputation:
    self.pipeline.null_imputation()  # ❌ Too late!

if apply_categorical_encoding:
    self.pipeline.encode_categoricals()  # ❌ Too late!
```

**CORRECT ORDER (change to)**:
```python
if apply_null_imputation:
    self.pipeline.null_imputation()  # ✅ Step 1: Fill missing MAAS

if apply_categorical_encoding:
    self.pipeline.encode_categoricals()  # ✅ Step 2: Encode categories

if apply_binning:
    self.pipeline.binning()  # ✅ Step 3: WOE binning (remove use_existing!)
```

**Also enable encoding** (Cell ~10):
```python
apply_null_imputation = True        # Change from False
apply_categorical_encoding = True    # Change from False
```

---

## 🏃 **Run Training**

### Step 1: Backup & Clean
```bash
# Backup current models
mkdir Models/backup
cp Models/*.pkl Models/backup/

# Delete old binning (REQUIRED!)
rm Models/binning.pkl
```

### Step 2: Start Jupyter
```bash
jupyter notebook BOB_Scorecard_Training.ipynb
```

### Step 3: Run All Cells
- Click "Cell" → "Run All"
- Wait ~15-20 minutes
- Monitor for errors

---

## 📊 **Expected Results**

### Performance Improvement
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Base Model Gini | 0.563 | 0.58-0.61 | +0.02 to +0.05 |
| Meta Model Gini | 0.575 | 0.59-0.62 | +0.02 to +0.05 |

### Feature Rankings (Expected)
1-5: **MAAS** (salary)
5-15: **SAHƏLƏR** (business sector)
15-25: **WORKGROUP** (employment type)
20-30: **QRUP** (organization type)
...existing credit bureau features...

### Files Created
- `Models/base_model_training.pkl` (40 MB)
- `Models/good_model_training.pkl` (30 MB)
- `Models/not_good_model_training.pkl` (20 MB)
- `Models/meta_model_training.pkl` (5 MB)
- `Models/binning.pkl` (NEW - with new features)
- `Output/FEATURE_IMPORTANCES_BASE_training.xlsx` (check MAAS ranking!)
- `Output/BINNING_TABLE_BASE_training.xlsx` (verify new features present)
- `Output/MAIN_SCORES_training.xlsx` (final scores)

---

## ✅ **Success Indicators**

Look for these in notebook output:

1. **Encoding Started**:
   ```
   Encoding started...
   Encoding completed.
   ```

2. **WOE Binning**:
   ```
   Fitting WOE binning for 150 variables...
   WOE binning completed for 148 variables
   ```

3. **Feature Count**:
   ```
   Training data loaded: (85955, 2710)  # ~2710 features after drops
   ```

4. **Gini Improvement**:
   ```
   Test gini: 0.58+ (instead of 0.563)
   ```

---

## 🚨 **If Something Goes Wrong**

### Error: "Module 'binning' has no attribute 'OptimalBinning'"
**Fix**: Restart Jupyter kernel after replacing binning.py

### Error: "Column XYZ not found"
**Fix**: Check your data.csv has all 2,748 columns

### Error: "binning.pkl not found"
**Fix**: This is EXPECTED - you deleted it! New one will be created.

### Warning: "Encoding failed - all values numeric"
**Fix**: Ignore - means feature was already numeric (like MAAS)

---

## 📚 **Detailed Documentation**

- `RETRAINING_GUIDE.md` - Complete step-by-step guide with all details
- `CATEGORICAL_ENCODING_GUIDE.md` - Deep dive into TargetEncoder setup
- `explore_new_features.py` - Data exploration script (optional)

---

## 🎯 **Ready? Let's Go!**

1. ✅ Make 2 changes in notebook (cols_to_drop + pipeline order)
2. ✅ Backup models + delete binning.pkl
3. ✅ Run All Cells in Jupyter
4. ✅ Wait ~15-20 minutes
5. ✅ Check Gini improvement: 0.563 → 0.58+
6. ✅ Celebrate! 🎉

**Questions?** Check RETRAINING_GUIDE.md or CATEGORICAL_ENCODING_GUIDE.md for details!
