# CORRECTED PIPELINE CLASSES - Copy to Notebook
# Apply these changes to BOB_Scorecard_Training.ipynb

## PIPELINE SETTINGS (Use for ALL layers)
apply_data_explore = True
apply_null_elimination = True
apply_constant_elimination = True
apply_null_imputation = True          # BEFORE binning
apply_binning = True                   # PRIMARY encoding method
apply_low_gini_elimination = True      # AFTER binning
apply_correlated_feature_elimination = True
apply_scaling = False                  # For Layer 1 (Base)
apply_categorical_encoding = False     # SKIP - binning handles encoding!

# For Layer 2, enable scaling:
# apply_scaling = True

apply_LR = True
apply_RF = True   # Layer 2 only
apply_XGB = True  # Layer 2 only
apply_LGBM = True # Layer 2 only


# ============================================================================
# LAYER 1: BASE MODEL
# ============================================================================

class Pipeline_Base:
    """
    Base model pipeline with corrected order
    """

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
        # BEFORE binning - handle missing values first

        if apply_null_imputation:
            self.pipeline.null_imputation()


        ########################   Binning (WOE Encoding)    ############################
        # PRIMARY encoding method - handles both numeric and categorical features
        # MOVED HERE: Before elimination so all features can be evaluated

        if apply_binning:
            self.pipeline.binning()


        ########################   Low Gini Elimination    ############################
        # NOW runs AFTER binning, so all features are WOE-encoded (numeric)

        if apply_low_gini_elimination:
            self.pipeline.feature_elimination(eliminator='drop_low_gini_features', params={'threshold':0.05, 'missing_values':'ignore'})


        ########################   Feature Elimination    ############################

        if apply_correlated_feature_elimination:
            self.pipeline.feature_elimination(eliminator='correlated_lower_gini_feature_elimination', params={'missing_values':'ignore'})


        ########################   Scaling    ############################

        if apply_scaling:
            self.pipeline.scaling()


        ########################   Categorical Encoding    ############################
        # SKIP THIS - binning already encoded everything with WOE
        # Only use if binning is disabled

        # if apply_categorical_encoding and not apply_binning:
        #     self.pipeline.encode_categoricals()


        #########################      Logistic Regression      #############################

        if apply_LR:
            self.pipeline.LogisticRegression(param_opt=True, param_opt_method = "RandomizedSearchCV", train_size=x_train.shape[0], param_grid = param_grid_best['logistic_regression_base'])


    def apply(self, x_test, y_test=[]):
        predicts = self.pipeline.test(test = x_test, y_test=y_test)
        return predicts


# ============================================================================
# LAYER 2a: NOT GOOD MODEL
# ============================================================================

class Pipeline_Not_Good:
    """
    Not Good segment pipeline - SAME ORDER as Base
    """

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
        # BEFORE binning

        if apply_null_imputation:
            self.pipeline.null_imputation()


        ########################   Binning (WOE Encoding)    ############################
        # PRIMARY encoding - BEFORE elimination

        if apply_binning:
            self.pipeline.binning()


        ########################   Low Gini Elimination    ############################
        # AFTER binning

        if apply_low_gini_elimination:
            self.pipeline.feature_elimination(eliminator='drop_low_gini_features', params={'threshold':0.05, 'missing_values':'ignore'})


        ########################   Feature Elimination    ############################

        if apply_correlated_feature_elimination:
            self.pipeline.feature_elimination(eliminator='correlated_lower_gini_feature_elimination', params={'missing_values':'ignore'})


        ########################   Scaling    ############################
        # For Layer 2, scaling is typically enabled

        if apply_scaling:
            self.pipeline.scaling()


        ########################   Categorical Encoding    ############################
        # SKIP - binning handles it

        # if apply_categorical_encoding and not apply_binning:
        #     self.pipeline.encode_categoricals()


        #########################      Logistic Regression      #############################

        if apply_LR:
            self.pipeline.LogisticRegression(param_opt=True, param_opt_method = "RandomizedSearchCV", train_size=x_train.shape[0], param_grid = param_grid_best['logistic_regression_not_good'])

        #########################      Random Forest      #############################

        if apply_RF:
            self.pipeline.RandomForest(param_opt=True, param_opt_method = "RandomizedSearchCV", train_size=x_train.shape[0], param_grid = param_grid_best['random_forest_not_good'])

        #########################        XGBoost          #############################

        if apply_XGB:
            self.pipeline.XGBoost(param_opt=True, param_opt_method = "RandomizedSearchCV", train_size=x_train.shape[0], param_grid = param_grid_best['xgboost_not_good'])

        #########################      LGBM      #############################

        if apply_LGBM:
            self.pipeline.LGBM(param_opt=True, param_opt_method = "RandomizedSearchCV", train_size=x_train.shape[0], param_grid = param_grid_best['lightgbm_not_good'])

    def apply(self, x_test, y_test=[]):
        predicts = self.pipeline.test(test = x_test, y_test=y_test)
        return predicts


# ============================================================================
# LAYER 2b: GOOD MODEL
# ============================================================================

class Pipeline_Good:
    """
    Good segment pipeline - SAME ORDER as Base
    """

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
        # BEFORE binning

        if apply_null_imputation:
            self.pipeline.null_imputation()


        ########################   Binning (WOE Encoding)    ############################
        # PRIMARY encoding - BEFORE elimination

        if apply_binning:
            self.pipeline.binning()


        ########################   Low Gini Elimination    ############################
        # AFTER binning

        if apply_low_gini_elimination:
            self.pipeline.feature_elimination(eliminator='drop_low_gini_features', params={'threshold':0.05, 'missing_values':'ignore'})


        ########################   Feature Elimination    ############################

        if apply_correlated_feature_elimination:
            self.pipeline.feature_elimination(eliminator='correlated_lower_gini_feature_elimination', params={'missing_values':'ignore'})


        ########################   Scaling    ############################
        # For Layer 2, scaling is typically enabled

        if apply_scaling:
            self.pipeline.scaling()


        ########################   Categorical Encoding    ############################
        # SKIP - binning handles it

        # if apply_categorical_encoding and not apply_binning:
        #     self.pipeline.encode_categoricals()


        #########################      Logistic Regression      #############################

        if apply_LR:
            self.pipeline.LogisticRegression(param_opt=True, param_opt_method = "RandomizedSearchCV", train_size=x_train.shape[0], param_grid = param_grid_best['logistic_regression_good'])

        #########################      Random Forest      #############################

        if apply_RF:
            self.pipeline.RandomForest(param_opt=True, param_opt_method = "RandomizedSearchCV", train_size=x_train.shape[0], param_grid = param_grid_best['random_forest_good'])

        #########################        XGBoost          #############################

        if apply_XGB:
            self.pipeline.XGBoost(param_opt=True, param_opt_method = "RandomizedSearchCV", train_size=x_train.shape[0], param_grid = param_grid_best['xgboost_good'])

        #########################      LGBM      #############################

        if apply_LGBM:
            self.pipeline.LGBM(param_opt=True, param_opt_method = "RandomizedSearchCV", train_size=x_train.shape[0], param_grid = param_grid_best['lightgbm_good'])

    def apply(self, x_test, y_test=[]):
        predicts = self.pipeline.test(test = x_test, y_test=y_test)
        return predicts


# ============================================================================
# LAYER 3: META MODEL
# ============================================================================

class Pipeline_Meta:
    """
    Meta model pipeline - No preprocessing needed, just model training
    Features are already probabilities from Layer 1 and Layer 2
    """

    def train(self, x_train, y_train, x_test=[], y_test=[]):
        if (len(x_test) > 0) & (len(y_test) > 0):
            x_val = x_test.copy()
            y_val = y_test.copy()

        self.pipeline = skills_api.ClassifierPipeline(x_train, y_train, x_val, y_val)

        ########################   Data Explore    ############################

        if apply_data_explore:
            self.pipeline.data_explore()


        # NO PREPROCESSING for meta model - features are already probabilities
        # Skip: null elimination, constant elimination, imputation, binning, elimination


        #########################      Logistic Regression      #############################

        if apply_LR:
            self.pipeline.LogisticRegression(param_opt=True, param_opt_method = "RandomizedSearchCV", train_size=x_train.shape[0], param_grid = param_grid_best['logistic_regression_meta'])


    def apply(self, x_test, y_test=[]):
        predicts = self.pipeline.test(test = x_test, y_test=y_test)
        return predicts


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
STEP 1: Update Pipeline Settings

For Layer 1 (Base):
    apply_scaling = False
    apply_binning = True
    apply_categorical_encoding = False

For Layer 2 (Not Good / Good):
    apply_scaling = True
    apply_binning = True
    apply_categorical_encoding = False

For Layer 3 (Meta):
    All preprocessing flags = False (no preprocessing needed)
    Only apply_LR = True


STEP 2: Replace Pipeline Classes

In your notebook:
1. Find each pipeline class definition
2. Replace with the corrected version above
3. Ensure settings match (scaling enabled for Layer 2)


STEP 3: Re-run Training

After replacing all classes:
1. Re-run Layer 1 training (should work as before)
2. Re-run Layer 2 training (should fix performance issues)
3. Re-run Layer 3 training (should maintain good performance)


EXPECTED IMPROVEMENTS:
- XGBoost in Good model should work (no longer Gini=0)
- Better validation performance in Layer 2
- Consistent WOE encoding across all layers
- No more double encoding issues
"""
