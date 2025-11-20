# Instructions for Running Original Credit Scoring Notebooks with CSV/XLSX Data

## Overview

This guide shows how to run the **original 2 notebooks** using CSV/XLSX files instead of Oracle database.

---

## 📁 Files You'll Use

### Original Notebooks (Main Workflow)
1. **BOB_Scorecard_Training.ipynb** - Train 3-layer model
2. **BOB_Scorecard_Scoring_Policy_Adjustment.ipynb** - Score new applicants

### Files to Ignore
- BOB_Scorecard_Unified_Pipeline.ipynb (newer unified version)
- generate_sample_data.py (sample data generator)
- test_pipeline_with_sample_data.py (test script)

---

## 🔧 Setup Instructions

### Step 1: Prepare Your Data

Create CSV or XLSX files with this structure:

**train_data.csv:**
```csv
MUQAVILE,TARGET,CC_O_3MWPS_EVER,CL_A_CWPS_90D,AGE,INCOME,...
CONT_001,0,0,1,42,5000,...
CONT_002,1,2,3,35,3500,...
CONT_003,0,0,0,51,6200,...
```

**Required Columns:**
- `MUQAVILE` - Contract ID (unique identifier)
- `TARGET` - Binary target (0=good, 1=default)
- Feature columns - Credit bureau, demographics, application features

**Save files to:**
```
Data/
├── train_data.csv       # Training data
├── test_data.csv        # Test/holdout data
└── score_data.csv       # New applicants to score
```

---

## 📝 Notebook Modifications

### Modify BOB_Scorecard_Training.ipynb

#### Cell 9 - Comment out sorted_indices (optional reproduction)
```python
# ############ (OPTIONAL) DATA ORDER IN THE FINAL VERSION ############
# sorted_indices = pd.read_excel("Data/sorted_indices.xlsx")
# sorted_indices_train = sorted_indices.iloc[:85955,0].values
# sorted_indices_test = sorted_indices.iloc[85955:,0].values
```

#### Cell 10 - Replace database loading with CSV/XLSX reading

**Original (database):**
```python
############ READ DATA FROM DATABASE ############
database_username = pd.read_table('Data/user', header = None)[0][0]
database_password = pd.read_table('Data/pass', header = None)[0][0]
engine=data.connect_to_sql(database_username, database_password)

with open(f'Data/train_data_sql_{version}.txt') as f: train_data_sql = f.read()
train = data.data_load(engine, sql = train_data_sql)

train = train.set_index(index_col).rename(columns={target_col: "TARGET"})
train = train.reindex(sorted_indices_train)
y_train = train["TARGET"]
x_train = train.drop(columns=["TARGET"]+cols_to_drop, inplace=False)

############ READ TEST (HOLDOUT) DATA ############
engine=data.connect_to_sql(database_username, database_password)
with open(f'Data/test_data_sql_{version}.txt') as f:  test_data_sql = f.read()
test = data.data_load(engine, sql = test_data_sql)

test = test.set_index(index_col).rename(columns={target_col: "TARGET"})
test = test.reindex(sorted_indices_test)
y_test = test["TARGET"]
x_test = test.drop(columns=["TARGET"]+cols_to_drop, inplace=False)
```

**New (CSV/XLSX):**
```python
############ READ DATA FROM CSV/XLSX ############

# Option 1: Read from CSV
train = pd.read_csv('Data/train_data.csv')

# Option 2: Read from Excel (uncomment if using XLSX)
# train = pd.read_excel('Data/train_data.xlsx')

train = train.set_index(index_col).rename(columns={target_col: "TARGET"})
# Comment out if not using sorted_indices
# train = train.reindex(sorted_indices_train)
y_train = train["TARGET"]
x_train = train.drop(columns=["TARGET"]+cols_to_drop, inplace=False)

print(f"Training data loaded: {x_train.shape}")
print(f"Default rate: {y_train.mean():.2%}")


############ READ TEST (HOLDOUT) DATA ############

# Option 1: Read from CSV
test = pd.read_csv('Data/test_data.csv')

# Option 2: Read from Excel (uncomment if using XLSX)
# test = pd.read_excel('Data/test_data.xlsx')

test = test.set_index(index_col).rename(columns={target_col: "TARGET"})
# Comment out if not using sorted_indices
# test = test.reindex(sorted_indices_test)
y_test = test["TARGET"]
x_test = test.drop(columns=["TARGET"]+cols_to_drop, inplace=False)

print(f"Test data loaded: {x_test.shape}")
print(f"Default rate: {y_test.mean():.2%}")
```

---

### Modify BOB_Scorecard_Scoring_Policy_Adjustment.ipynb

Find the data loading cell and replace with:

```python
############ READ SCORING DATA FROM CSV/XLSX ############

# Option 1: Read from CSV
new_data = pd.read_csv('Data/score_data.csv')

# Option 2: Read from Excel (uncomment if using XLSX)
# new_data = pd.read_excel('Data/score_data.xlsx')

new_data = new_data.set_index(index_col)

# Case 1: If you have TARGET column (for validation)
if 'TARGET' in new_data.columns:
    new_data = new_data.rename(columns={target_col: "TARGET"})
    y_new = new_data["TARGET"]
    x_new = new_data.drop(columns=["TARGET"]+cols_to_drop, inplace=False)
    print(f"Scoring data with target: {x_new.shape}")
# Case 2: No TARGET column (real production scoring)
else:
    x_new = new_data.drop(columns=cols_to_drop, errors='ignore', inplace=False)
    y_new = None
    print(f"Scoring data (no target): {x_new.shape}")
```

---

## 🚀 Execution Steps

### Phase 1: Training (BOB_Scorecard_Training.ipynb)

**Run cells in order:**

1. **Cells 1-6:** Environment setup and library imports
2. **Cell 7:** Set parameters
   ```python
   version = "training"
   index_col = "MUQAVILE"
   target_col = "TARGET"
   ```

3. **Cell 8:** Define cols_to_drop
4. **Cell 9:** (Comment out sorted_indices if not using)
5. **Cell 10:** Load data from CSV/XLSX (modified cell)

6. **Cells 11-14:** Train Layer 1 (Base Model)
   - WOE binning + Logistic Regression
   - Saves to `Models/base_model_training.pkl`

7. **Cells 15-17:** Split data for Layer 2
   - Good segment: score < 180
   - Not-Good segment: score >= 200

8. **Cells 18-24:** Train Layer 2 (Segment Models)
   - Good model: LightGBM
   - Not-Good model: Logistic Regression
   - Saves to `Models/good_model_training.pkl` and `Models/not_good_model_training.pkl`

9. **Cells 25-30:** Train Layer 3 (Meta Model)
   - Combines all predictions
   - Saves to `Models/meta_model_training.pkl`

10. **Cell 31:** Calculate final scores
    - Saves to `Output/MAIN_SCORES_training.xlsx`

**Expected Runtime:** 10-15 minutes (depending on data size)

**Expected Outputs:**
```
Models/
├── base_model_training.pkl
├── good_model_training.pkl
├── not_good_model_training.pkl
└── meta_model_training.pkl

Output/
├── MAIN_SCORES_training.xlsx
├── FEATURE_IMPORTANCES_BASE_training.xlsx
├── BINNING_TABLE_BASE_training.xlsx
└── ... (various elimination feature files)
```

---

### Phase 2: Scoring (BOB_Scorecard_Scoring_Policy_Adjustment.ipynb)

**Prerequisites:**
- Models trained from Phase 1
- New applicant data in `Data/score_data.csv`

**Run cells in order:**

1. **Cells 1-5:** Environment setup and library imports
2. **Cell 6:** Set version and parameters
   ```python
   version = "training"  # Must match training version
   index_col = "MUQAVILE"
   ```

3. **Cell 7:** Load new data from CSV/XLSX (modified cell)

4. **Cell 8:** Load trained models
   ```python
   with open(f'Models/base_model_{version}.pkl', 'rb') as f:
       pipeline_base = pickle.load(f)
   with open(f'Models/good_model_{version}.pkl', 'rb') as f:
       pipeline_good = pickle.load(f)
   with open(f'Models/not_good_model_{version}.pkl', 'rb') as f:
       pipeline_not_good = pickle.load(f)
   with open(f'Models/meta_model_{version}.pkl', 'rb') as f:
       pipeline_meta = pickle.load(f)
   ```

5. **Cells 9-15:** Apply pipeline and calculate scores
   - Layer 1 predictions
   - Layer 2 predictions (good/not-good)
   - Layer 3 meta predictions
   - Convert probability to score
   - Apply policy adjustment (cap at 250, discount 5%)

6. **Final Cell:** Save results
   - Saves to `Output/SCORING_RESULTS_training.xlsx`

**Expected Runtime:** 2-5 minutes

**Expected Output:**
```
Output/
└── SCORING_RESULTS_training.xlsx
    Columns: MUQAVILE | PROBA | RAW_SCORE | FINAL_SCORE | [TARGET]
```

---

## 📊 Understanding the Outputs

### MAIN_SCORES_training.xlsx (from training)

| Column | Description |
|--------|-------------|
| MUQAVILE | Contract ID (index) |
| BASIC_PROBA | Layer 1 probability (base model only) |
| BASIC_SCORE | Layer 1 score |
| BOOSTED_PROBA | Final probability (all 3 layers) |
| BOOSTED_SCORE | Final score (all 3 layers) |
| TARGET | Actual outcome (0=good, 1=default) |

### SCORING_RESULTS_training.xlsx (from scoring)

| Column | Description |
|--------|-------------|
| MUQAVILE | Contract ID (index) |
| PROBA | Final default probability (0-1) |
| RAW_SCORE | Credit score before policy adjustment |
| FINAL_SCORE | Final score after cap (250) and discount (×0.95) |
| TARGET | Actual outcome if available (optional) |

**Score Interpretation:**
- 220-250: Excellent (< 0.5% default)
- 200-220: Good (0.5-1% default)
- 180-200: Fair (1-3% default)
- 160-180: Poor (3-6% default)
- < 160: High Risk (> 6% default)

---

## 🎯 Key Parameters

### Credit Score Transformation

**Layer 1 (Base Model) - Used for segmentation:**
```python
ref = 200
odds_at_ref = 50        # For segmentation thresholds
points_to_double = 20
```

**Layer 3 (Meta Model) - Used for final scoring:**
```python
ref = 200
odds_at_ref = 100       # For final credit score
points_to_double = 20
```

**Segmentation Thresholds:**
```python
good_score_threshold = 180      # Good segment: < 180
not_good_score_threshold = 200  # Not-good segment: >= 200
```

**Policy Adjustment:**
```python
final_score = min(raw_score, 250, raw_score × 0.95)
```

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'QNBAnalytics_ML'"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Error: "FileNotFoundError: Data/train_data.csv"
**Solution:** Ensure CSV files are in the Data/ directory

### Error: "KeyError: 'TARGET'"
**Solution:** Check that your CSV has a TARGET column (or adjust target_col parameter)

### Error: "ValueError: could not convert string to float"
**Solution:** Ensure all feature columns are numeric (no text except in MUQAVILE/TARGET)

### Models not loading in scoring notebook
**Solution:** Ensure `version` parameter matches between training and scoring notebooks

---

## 📌 Quick Reference

**Training Workflow:**
```
CSV/XLSX → Load Data → Train 3 Layers → Save Models & Scores
```

**Scoring Workflow:**
```
CSV/XLSX → Load Data → Load Models → Apply Pipeline → Save Scores
```

**Files Modified:**
- BOB_Scorecard_Training.ipynb (Cell 9, Cell 10)
- BOB_Scorecard_Scoring_Policy_Adjustment.ipynb (Data loading cell)

**No Changes Needed:**
- QNBAnalytics_ML/ library
- Model training logic
- Scoring logic

---

## 💡 Tips

1. **Start with small data** (~1,000 rows) to test the workflow
2. **Check data types** - all features should be numeric
3. **Monitor Gini scores** - Layer 1 should be ~0.55-0.60, Layer 3 should be ~0.57-0.62
4. **Keep version consistent** - Use same `version` parameter in training and scoring
5. **Backup original notebooks** - Keep a copy before making changes

---

For questions, refer to README.md or contact the QNBAnalytics ML Team.
