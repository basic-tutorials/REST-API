# Pipeline Order Fix - Remove Double Encoding

## Problem
Currently categorical features are being encoded TWICE:
1. First by `encode_categoricals()` (TargetEncoder)
2. Then by `binning()` (WOEEncoder)

This destroys the predictive signal and causes inconsistencies.

## Solution
Use **ONLY WOEEncoder** by moving binning BEFORE elimination.

---

## Changes Needed in Notebook

### Current Pipeline (Cell: Pipeline_Base.train())

```python
########################   Null Imputation    ############################

if apply_null_imputation:
    self.pipeline.null_imputation()


########################   Categorical Encoding    ############################

if apply_categorical_encoding:
    self.pipeline.encode_categoricals()


########################   Low Gini Elimination    ############################

if apply_low_gini_elimination:
    self.pipeline.feature_elimination(eliminator='drop_low_gini_features', params={'threshold':0.05, 'missing_values':'ignore'})


########################   Feature Elimination    ############################

if apply_correlated_feature_elimination:
    self.pipeline.feature_elimination(eliminator='correlated_lower_gini_feature_elimination', params={'missing_values':'ignore'})


########################   Binning    ############################

if apply_binning:
    self.pipeline.binning()
```

### NEW Pipeline Order (APPLY THIS)

```python
########################   Null Imputation    ############################

if apply_null_imputation:
    self.pipeline.null_imputation()


########################   Binning    ############################
# MOVED HERE! Now handles ALL encoding (numeric + categorical) with WOE

if apply_binning:
    self.pipeline.binning()  # WOEEncoder handles both numeric and categorical


########################   Low Gini Elimination    ############################
# Now happens AFTER WOE encoding, so all features are numeric

if apply_low_gini_elimination:
    self.pipeline.feature_elimination(eliminator='drop_low_gini_features', params={'threshold':0.05, 'missing_values':'ignore'})


########################   Feature Elimination    ############################

if apply_correlated_feature_elimination:
    self.pipeline.feature_elimination(eliminator='correlated_lower_gini_feature_elimination', params={'missing_values':'ignore'})


########################   Categorical Encoding    ############################
# SKIP THIS! binning() already encoded everything
# Only needed if apply_binning=False

# if apply_categorical_encoding and not apply_binning:  # Only if binning is disabled
#     self.pipeline.encode_categoricals()
```

---

## Full Corrected Pipeline Order

```python
class Pipeline_Base:
    def train(self, x_train, y_train, x_test=[], y_test=[]):
        if (len(x_test) > 0) & (len(y_test) > 0):
            x_val = x_test.copy()
            y_val = y_test.copy()

        self.pipeline = skills_api.ClassifierPipeline(x_train, y_train, x_val, y_val)

        ########################   Data Explore    ############################

        if apply_data_explore:
            self.pipeline.data_explore()


        ########################   Drop Null Features    ############################

        if apply_null_elimination:
            self.pipeline.feature_elimination(eliminator='drop_null_features', params={'threshold': 0.99})


        ########################   Drop Constant Features    ############################

        if apply_constant_elimination:
            self.pipeline.feature_elimination(eliminator='drop_constant_features', params={'missing_values':'ignore'})


        ########################   Null Imputation    ############################

        if apply_null_imputation:
            self.pipeline.null_imputation()


        ########################   Binning (WOE Encoding for ALL features)    ############################
        # THIS IS NOW THE PRIMARY ENCODING METHOD
        # Handles numeric (with binning first) + categorical (directly)

        if apply_binning:
            self.pipeline.binning()


        ########################   Low Gini Elimination    ############################
        # Now all features are WOE-encoded (numeric), so Gini can be calculated

        if apply_low_gini_elimination:
            self.pipeline.feature_elimination(eliminator='drop_low_gini_features', params={'threshold':0.05, 'missing_values':'ignore'})


        ########################   Feature Elimination    ############################

        if apply_correlated_feature_elimination:
            self.pipeline.feature_elimination(eliminator='correlated_lower_gini_feature_elimination', params={'missing_values':'ignore'})


        ########################   Scaling    ############################

        if apply_scaling:
            self.pipeline.scaling()


        #########################      Logistic Regression      #############################

        if apply_LR:
            self.pipeline.LogisticRegression(param_opt=True, param_opt_method = "RandomizedSearchCV", train_size=x_train.shape[0], param_grid = param_grid_best['logistic_regression_base'])


    def apply(self, x_test, y_test=[]):
        predicts = self.pipeline.test(test = x_test, y_test=y_test)
        return predicts
```

---

## Update Settings

```python
## PIPELINE SETTINGS

apply_data_explore = True
apply_null_elimination = True
apply_constant_elimination = True
apply_null_imputation = True      # BEFORE binning - correct!
apply_binning = True               # PRIMARY encoding method
apply_low_gini_elimination = True  # AFTER binning - correct!
apply_correlated_feature_elimination = True
apply_scaling = False
apply_categorical_encoding = False  # SET TO FALSE - binning handles it!

apply_LR = True
```

---

## Expected Results After This Fix

✅ **No more double encoding**
- Categorical features encoded ONCE with WOEEncoder
- Numeric features binned, then WOE-encoded

✅ **Real Gini scores**
- Calculated from actual AUC during binning
- Displayed in GINI_TABLE_BASE_training.xlsx

✅ **New features should be retained**
- MAAS, WORKGROUP, QRUP, SAHƏLƏR, SEKTOR will:
  - Get proper WOE encoding
  - Get real Gini scores
  - Only be eliminated if Gini < 0.05 (legitimately low predictive power)

✅ **Binning table will have real WOE values**
- Not all zeros
- Proper WOE values for each bin/category

---

## Apply These Changes

1. Open `BOB_Scorecard_Training.ipynb`
2. Find the `Pipeline_Base` class definition
3. Rearrange the order as shown above
4. Set `apply_categorical_encoding = False`
5. Re-run the training cells

---

## Validation Checklist

After running:
- [ ] Check `Output/BINNING_TABLE_BASE_training.xlsx` - WOE values should NOT be all zeros
- [ ] Check `Output/GINI_TABLE_BASE_training.xlsx` - Gini scores should be between 0 and 1
- [ ] Check `Output/FEATURE_IMPORTANCES_BASE_training.xlsx` - Look for MAAS, WORKGROUP, QRUP, SAHƏLƏR, SEKTOR
- [ ] Check `Output/GINI_ELIMINATED_FEATURES_BASE_training.xlsx` - See which new features were eliminated and why
