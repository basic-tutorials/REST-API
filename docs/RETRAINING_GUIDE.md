# 🔄 Model Retraining Guide - Including New Features

## ✅ **Completed Steps**

### 1. Replaced OptBinning Library
- ❌ **OLD**: `optbinning>=0.17.0` (had compatibility issues with Python 3.8.16)
- ✅ **NEW**: Custom WOE binning implementation in `QNBAnalytics_ML/binning.py`
- **Reason**: optbinning and scorecardpy both require scipy which needs a C compiler on Windows
- **Solution**: Created lightweight custom WOE binning using sklearn DecisionTree (already installed)

### 2. Backed Up Files
- ✅ `QNBAnalytics_ML/binning_OLD_BACKUP.py` - Original binning code
- 📂 Next: Backup Models folder before retraining

### 3. Identified New Features
Found **8 new columns** in data.csv:

| Column | Type | Use? | Reason |
|--------|------|------|--------|
| **MKR_ID** | ID | ❌ NO | ID field (not predictive) |
| **IDS** | ID | ❌ NO | Sequential ID (not predictive) |
| **MAAS** | Numeric | ✅ YES | Salary - strong income predictor |
| **WORKGROUP** | Categorical | ✅ YES | Employment type - stability indicator |
| **QRUP** | Categorical | ✅ YES | Organization type - job sector |
| **SAHƏLƏR** | Categorical | ✅ YES | Business sector - industry risk |
| **SEKTOR** | Categorical | ⚠️ MAYBE | Specific sector - may be redundant with SAHƏLƏR |
| **PARTNYORLUQ** | Categorical | ⚠️ MAYBE | Partnership - check value distribution first |

---

## 📋 **Required Changes to Training Notebook**

### Change 1: Update `cols_to_drop` (Cell 2)

**Location**: Cell with ID `6f251bf9-b9dc-4d21-8644-c3a3c9046ea1`

**BEFORE**:
```python
cols_to_drop = ["CC_O_4M6MWPS_EVER","ALL_O_4M6MWPS_EVER_O",...,"BGN"]
```

**AFTER**:
```python
############ ENTER PARAMETERS ############

index_col = "MUQAVILE"  # index column
target_col = "TARGET"  # target column

## Columns to drop
cols_to_drop = [
    # Original dropped features (KEEP THESE)
    "CC_O_4M6MWPS_EVER", "ALL_O_4M6MWPS_EVER_O", "ALL_O_7M12MWPS_EVER",
    "ALL_O_7M12MWPS_EVER_O", "ALL_OSMLMT_CWPS1_6_EVER", "CC_O_13M24MWPS_EVER",
    "CC_O_13M24MWPS_EVER_O", "CC_O_7M12MWPS_EVER", "CC_O_7M12MWPS_EVER_O",
    "CCOL_O_4M6MWPS_EVER", "CL_O_13M24MWPS_EVER_O", "HL_EVERWPS_EVER",
    "HL_O_13M24MWPS_365DP", "HL_O_3MWPS_365DP", "HL_O_4M6MWPS_EVER_O",
    "HL_O_7M12MWPS_EVER_O", "HL_O_EVERWPS_365DP", "OL_O_3MWPS_183D365D",
    "OL_O_3MWPS_365DP", "OL_O_4M6MWPS_183D365D", "OL_O_4M6MWPS_91D182D",
    "OL_O_7M12MWPS_365DP", "OL_O_EVERWPS_183D365D", "OL_O_EVERWPS_365DP",
    "OL_O_EVERWPS_90D", "OL_O_EVERWPS_91D182D", "OL_OLMTUTL_CWPS0_90D_O",
    "BGN",

    # NEW: Add ID and non-predictive columns
    "ID", "FIN", "MKR_ID", "MKR_DATE", "IDS",
    "SCR_CONSTANT", "SCR_MULTIPLIER",

    # OPTIONAL: Uncomment after exploration if not useful
    # "PARTNYORLUQ",  # If low variation
    # "SEKTOR",  # If too redundant with SAHƏLƏR
]
```

---

### Change 2: Enable Categorical Encoding & Imputation (Cell ~10)

**Location**: Cell with ID `dc648bc2-d752-40f9-a873-3da198406c48`

**BEFORE**:
```python
apply_null_imputation = False
apply_categorical_encoding = False
```

**AFTER**:
```python
## PIPELINE SETTINGS FOR LAYER 1 (BASE MODEL)

apply_data_explore = True
apply_null_elimination = True
apply_constant_elimination = True
apply_low_gini_elimination = True
apply_correlated_feature_elimination = True
apply_binning = True
apply_scaling = False
apply_null_imputation = True        # ✅ CHANGED: Handle missing MAAS values
apply_categorical_encoding = True    # ✅ CHANGED: Encode new categorical features

apply_LR = True
```

---

### Change 3: Update Pipeline_Base Class (Cell ~11)

**Location**: Cell with ID `11045bb8-eae1-4ab6-bf10-2717dacbfb49`

**CRITICAL CHANGES**:
1. Move `null_imputation` BEFORE `binning`
2. Move `categorical_encoding` BEFORE `binning`
3. Remove `use_existing = 'binning.pkl'` from binning call

**Find this section in Pipeline_Base.train():**
```python
########################   Feature Elimination    ############################

if apply_correlated_feature_elimination:
    self.pipeline.feature_elimination(eliminator='correlated_lower_gini_feature_elimination', params={'missing_values':'ignore'})


########################   Binning    ############################

if apply_binning:
    self.pipeline.binning(use_existing = 'binning.pkl')  # ❌ REMOVE use_existing!


########################   Scaling    ############################

if apply_scaling:
    self.pipeline.scaling()


########################   Null Imputation    ############################

if apply_null_imputation:
    self.pipeline.null_imputation()


########################   Categorical Encoding    ############################

if apply_categorical_encoding:
    self.pipeline.encode_categoricals()
```

**CHANGE TO THIS ORDER**:
```python
########################   Feature Elimination    ############################

if apply_correlated_feature_elimination:
    self.pipeline.feature_elimination(eliminator='correlated_lower_gini_feature_elimination', params={'missing_values':'ignore'})


########################   Null Imputation    ############################
# ✅ MOVED BEFORE BINNING - Must impute numeric features (MAAS) first

if apply_null_imputation:
    self.pipeline.null_imputation()


########################   Categorical Encoding    ############################
# ✅ MOVED BEFORE BINNING - Must encode categorical features first

if apply_categorical_encoding:
    self.pipeline.encode_categoricals()


########################   Binning    ############################
# ✅ CHANGED: Removed use_existing to retrain binning with new features

if apply_binning:
    self.pipeline.binning()  # Retrain binning from scratch!


########################   Scaling    ############################

if apply_scaling:
    self.pipeline.scaling()
```

---

## 🚀 **Step-by-Step Retraining Process**

### Step 1: Backup Current Models ✅
```bash
mkdir Models/backup_$(date +%Y%m%d)
cp Models/*.pkl Models/backup_$(date +%Y%m%d)/
```

### Step 2: Delete Old Binning File ✅
```bash
rm Models/binning.pkl
```

### Step 3: Run Data Exploration (Optional but Recommended) 📊
```bash
python explore_new_features.py
```

This will show you:
- Default rates by salary quartiles
- Employment type distribution
- Sector-wise risk patterns
- Which features to keep/drop

### Step 4: Make the 3 Changes Above in Notebook 📝

Open `BOB_Scorecard_Training.ipynb` and:
1. ✅ Update `cols_to_drop` (add ID columns)
2. ✅ Enable `apply_null_imputation` and `apply_categorical_encoding`
3. ✅ Reorder Pipeline_Base class (impute → encode → bin)

### Step 5: Execute Training 🎯

**Option A: Run in Jupyter** (Recommended)
```bash
jupyter notebook BOB_Scorecard_Training.ipynb
# Then: Run All Cells
```

**Option B: Run as Script** (If you have nbconvert)
```bash
jupyter nbconvert --to script BOB_Scorecard_Training.ipynb
python BOB_Scorecard_Training.py
```

**Expected Runtime**: ~15-20 minutes
**Expected Output**:
- 4 new model files in `Models/` (85-95 MB total)
- Output Excel files in `Output/` folder
- Feature importance tables showing new features
- Improved Gini score (baseline: 0.56 → expected: 0.58-0.61)

---

## 📊 **Expected Improvements**

### Feature Impact Predictions

| Feature | Expected Gini Impact | Expected Rank |
|---------|---------------------|---------------|
| **MAAS** | ⭐⭐⭐⭐⭐ High | Top 5-10 |
| **WORKGROUP** | ⭐⭐⭐ Medium | Top 20-30 |
| **QRUP** | ⭐⭐ Low-Medium | Top 50 |
| **SAHƏLƏR** | ⭐⭐⭐⭐ High | Top 15-25 |

### Model Performance

| Metric | Before | After (Expected) | Improvement |
|--------|--------|------------------|-------------|
| **Base Model Gini (Test)** | 0.563 | 0.580-0.610 | +0.02 to +0.05 |
| **Meta Model Gini (Test)** | 0.575 | 0.590-0.620 | +0.02 to +0.05 |
| **Training Time** | 13 min | 15-18 min | +2-5 min |

---

## ⚠️ **Common Issues & Solutions**

### Issue 1: "binning.pkl not found" Error
**Solution**: This is expected! You deleted it. The model will create a new one.

### Issue 2: "Column XYZ not found" Error
**Solution**: Check that your data.csv has all 2,748 columns. If not, regenerate from database.

### Issue 3: Encoding fails on categorical features
**Solution**: This means you have new categorical values in test set. The pipeline handles this automatically.

### Issue 4: Memory Error
**Solution**: Close other applications. data.csv is large (~87K rows x 2,748 cols).

### Issue 5: Import Error for binning
**Solution**: Restart Python kernel after replacing binning.py

---

## 📈 **Post-Training Validation**

### 1. Check Feature Importances
```bash
# Open Output/FEATURE_IMPORTANCES_BASE_training.xlsx
# Look for: MAAS, WORKGROUP, SAHƏLƏR in top 30 features
```

### 2. Verify Gini Improvement
```bash
# Check notebook output logs
# BASE MODEL:
#   Old: Train gini: 0.597, Test gini: 0.563
#   New: Train gini: 0.61-0.63, Test gini: 0.58-0.61
```

### 3. Review Binning Tables
```bash
# Open Output/BINNING_TABLE_BASE_training.xlsx
# Verify new features (MAAS, WORKGROUP, etc.) are present
# Check WOE values are reasonable (not all 0 or NaN)
```

### 4. Test Scoring Pipeline
```bash
# Open BOB_Scorecard_Scoring_Policy_Adjustment.ipynb
# Run with a small test dataset
# Verify no errors and scores are in 160-250 range
```

---

## 📁 **Files Modified**

✅ **QNBAnalytics_ML/binning.py** - Replaced with custom WOE implementation
✅ **requirements.txt** - Updated to remove optbinning
✅ **QNBAnalytics_ML/binning_OLD_BACKUP.py** - Backup of original
📝 **BOB_Scorecard_Training.ipynb** - Needs 3 changes (see above)
📝 **Models/binning.pkl** - Will be regenerated during training
📝 **Models/*.pkl** - Will be replaced with new models

---

## 🎓 **Understanding the New Features**

### MAAS (Salary)
- **Why Important**: Direct proxy for income stability
- **Risk Pattern**: Higher salary → Lower default rate
- **Expected**: Strong negative correlation with TARGET

### WORKGROUP (Employment Type)
- **Why Important**: Full-time vs part-time stability
- **Risk Pattern**: "əsas işçi heyəti" (main staff) → Lower risk
- **Expected**: Categorical predictor with 2-3 levels

### QRUP (Organization Type)
- **Why Important**: State sector = stable employment
- **Risk Pattern**: "Dövlət Sektoru" (State Sector) → Lower risk
- **Expected**: Binary or low-cardinality categorical

### SAHƏLƏR (Business Sector)
- **Why Important**: Industry-specific risk (e.g., oil/gas stability)
- **Risk Pattern**: Varies by sector (some high-risk, some low-risk)
- **Expected**: High-cardinality categorical (15-30 unique sectors)

---

## 🔍 **Next Steps After Retraining**

1. **Model Validation**
   - Compare old vs new model performance on holdout set
   - Check lift curves and KS statistics
   - Verify no data leakage from new features

2. **Production Deployment**
   - Update scoring API to handle new features
   - Retrain scoring pipeline with new models
   - Update SQL queries to include MAAS, WORKGROUP, QRUP, SAHƏLƏR

3. **Monitoring**
   - Track feature drift for new columns
   - Monitor MAAS value distribution over time
   - Check for changes in WORKGROUP/QRUP distributions

4. **Documentation**
   - Update model documentation with new features
   - Document business logic for MAAS cutoffs
   - Create sector-specific risk profiles

---

## ✅ **Checklist Before Starting**

- [ ] Backed up current Models folder
- [ ] Deleted Models/binning.pkl
- [ ] Verified data.csv has 2,748 columns including new features
- [ ] Updated requirements.txt (already done)
- [ ] Replaced binning.py with custom implementation (already done)
- [ ] Made 3 changes to training notebook (see above)
- [ ] Jupyter environment is ready
- [ ] Have ~15-20 minutes for training to complete

---

**Ready to retrain? Open `BOB_Scorecard_Training.ipynb` and make the 3 changes above, then run all cells!**
