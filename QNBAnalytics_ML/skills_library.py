# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from QNBAnalytics_ML import data, preprocessing, elimination, classifier, binning
from QNBAnalytics_ML.preprocessing import impute_null
from QNBAnalytics_ML.helpers import getArgumentsList, getLogger
import inspect
import re

np.random.seed(48)


def get_feature_list(train_features, test):

    test_features = test.columns
    apply = pd.DataFrame()
    feature_lists = []
    for train_feature in train_features:
        if train_feature in test_features:
            apply[train_feature] = test[train_feature]
            feature_lists.append(train_feature)
        else:
            apply[train_feature] = np.nan

    return apply, feature_lists


class DropCols:

    def __init__(self,
                 cols_to_drop=[]
                 ):
        self.cols_to_drop = cols_to_drop

    def transform(self, df, y=None, sample='Test'):
        logger = getLogger()
        if sample == 'Apply':

            df = df.drop(columns=self.cols_to_drop, axis=1, errors="ignore")
            logger.info("Drop Columns applied...\n")

        else:

            df = df.drop(columns=self.cols_to_drop, axis=1, errors="ignore")

        return df


class data_explore:

    def __init__(self):
        self.variable_type=None
        self.cat_variables=None
        self.num_variables=None
        self.null_df=None

    def fit(self, train, y_train=None):
        train = train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

        self.variable_type=data.explore(train)
        self.cat_variables=self.variable_type.loc[self.variable_type['datatypes']=="categoric"].index
        self.num_variables=self.variable_type.loc[self.variable_type['datatypes']=="numeric"].index
        self.null_df = self.variable_type['missing_rate']
        self.train_features = train.columns

        corrs = []
        for col in self.train_features:
            try:
                corr = train[col].corr(y_train)
            except:
                corr = np.nan
            corrs.append( (col, corr) )

        self.targetcorr_df = pd.DataFrame(corrs, columns = ["variable", "correlation"]).set_index("variable")

        n_df = pd.DataFrame(self.null_df)

    def transform(self, train, y_train=None, sample='Test'):

        if sample == 'Apply' and hasattr(self,'train_features'):

            apply, feature_lists = get_feature_list(self.train_features, train)

            apply = apply.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
            for c in self.cat_variables:
                apply[c] = apply[c].astype(object)

            train[feature_lists] = apply[feature_lists]

        else:

            train = train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
            for c in self.cat_variables:
                train[c] = train[c].astype(object)

        return train


class conversion:

    def __init__(self):
        self.variable_type=None
        self.cat_variables=None
        self.user_cats=None
        self.user_nums=None

    def fit(self, num_variables, cat_variables, user_cats=None, user_nums=None):

        self.user_cats=user_cats
        self.user_nums=user_nums
        if self.user_cats is None:
            user_cats=[]
        if self.user_nums is None:
            user_nums=[]
        self.train_features = user_cats + user_nums

        self.num_variables, self.cat_variables = data.conversion(num_variables, cat_variables, self.user_cats, self.user_nums)

        return self.num_variables, self.cat_variables

    def transform(self, train, y=None, sample='Test'):

        if sample == 'Apply' and hasattr(self,'train_features'):

            apply, feature_lists = get_feature_list(self.train_features, train)

            if self.user_cats is not None:
                for cols in self.user_cats:
                    apply[cols] = apply[cols].astype(object)

            if self.user_nums is not None:
                for cols in self.user_nums:
                    apply[cols] = pd.to_numeric(apply[cols], errors="coerce")

            train[feature_lists] = apply[feature_lists]

        else:

            if self.user_cats is not None:
                for cols in self.user_cats:

                    train[cols]=train[cols].astype(object)

            if self.user_nums is not None:
                for cols in self.user_nums:

                    train[cols] = pd.to_numeric(train[cols], errors="coerce")

        return train


class null_imputation:

    def __init__(self,
                 null_thresh,
                 fill_value,
                 num_fill_value,
                 num_null,
                 cat_null,
                 num_strategy,
                 cat_strategy
                 ):
        self.null_thresh=null_thresh
        self.features_to_fix=None
        self.fill_value=fill_value
        self.num_fill_value = num_fill_value
        self.imp_cat=None
        self.num_fix=None
        self.cat_fix=None
        self.imp_num=None
        self.imp_cat=None
        self.num_null=num_null
        self.cat_null=cat_null
        self.num_strategy=num_strategy
        self.cat_strategy=cat_strategy

    def fit(self, train, null_df, num_variables, cat_variables):
        #Null filling thresholds
        self.features_to_fix = np.array(null_df[(null_df<self.null_thresh) & (null_df>=0.0)].index.values)
        self.features_to_drop = np.array(null_df[(null_df>self.null_thresh)].index.values)
        self.num_fix = [feature for feature in num_variables if feature in self.features_to_fix]
        self.cat_fix = [feature for feature in cat_variables if feature not in self.features_to_drop]
        self.train_features = train.columns

        if self.num_null==True:
            #imp_num=None
            if len(self.num_fix) > 0:
                self.imp_num=impute_null(strategy=self.num_strategy,fill_value=self.num_fill_value)
                self.imp_num.fit(train[self.num_fix])

        if self.cat_null==True:
            #imp_cat=None
            if len(self.cat_fix) > 0:
                self.imp_cat=impute_null(strategy=self.cat_strategy,fill_value=self.fill_value)
                self.imp_cat.fit(train[self.cat_fix])

    def transform(self, train, y_train=None, sample='Test'):

        if sample=='Apply' and hasattr(self,'train_features'):

            apply, feature_lists = get_feature_list(self.train_features, train)

            # drop high missing columns
            if len(self.features_to_drop) > 0:
                apply = apply.drop(columns=[self.features_to_drop])

            if self.imp_num != None:
                apply[self.num_fix] = self.imp_num.transform(apply[self.num_fix])

            if self.imp_cat != None:
                apply[self.cat_fix] = self.imp_cat.transform(apply[self.cat_fix])

            train[feature_lists] = apply[feature_lists]

        else:

            # drop high missing columns
            if len(self.features_to_drop) > 0:
                train = train.drop(columns=[self.features_to_drop])

            if self.imp_num!=None:
                train[self.num_fix] = self.imp_num.transform(train[self.num_fix])

            if self.imp_cat!=None:
                train[self.cat_fix] = self.imp_cat.transform(train[self.cat_fix])

        return train

class encode_categoricals:

    def __init__(self,
                 cat_fix,
                 method,
                 sample,
                 min_samples_leaf
                 ):
        self.cat_fix=cat_fix
        self.method=method
        self.sample=sample
        self.min_samples_leaf=min_samples_leaf
        self.encoder=None

    def fit(self, train, y):

        self.encoder = preprocessing.encode(method=self.method, sample=self.sample, min_samples_leaf=self.min_samples_leaf)
        self.encoder.fit(train, y, cols_to_enc = self.cat_fix)
        if self.method == "One Hot Encoder":
            self.train_features = list(train.columns)
            self.train_features_2 = list(set(self.train_features) - set(self.cat_fix))
            self.train_features_2 = self.train_features_2 + list(self.encoder.encoder.get_feature_names_out())
        else:
            self.train_features =  list(train.columns)

    def transform(self, train, y_train=None, sample='Test'):

        if sample == 'Apply' and hasattr(self,'train_features'):

            apply, feature_lists = get_feature_list(self.train_features, train)

            apply = self.encoder.transform(apply)

            if self.method == "One Hot Encoder":
                train[self.train_features_2] = apply[self.train_features_2]
            else:
                train[feature_lists] = apply[feature_lists]
        else:

            train = self.encoder.transform(train)

        return train


class scaling:

    def __init__(self,
                 type,
                 copy,
                 with_mean,
                 with_std,
                 feature_range
                 ):
        self.type=type
        self.copy=copy
        self.with_mean=with_mean
        self.with_std=with_std
        self.feature_range=feature_range
        self.scaler=None

    def fit(self, train):

        self.scaler=preprocessing.scale(type=self.type, copy=self.copy,with_mean=self.with_mean,with_std=self.with_std,feature_range=self.feature_range)
        self.scaler.fit(train)
        self.train_features = train.columns

    def transform(self, train, y_train=None, sample='Test'):

        if sample == 'Apply' and hasattr(self,'train_features'):

            apply, feature_lists = get_feature_list(self.train_features, train)

            apply[:] = self.scaler.transform(apply)

            train[feature_lists] = apply[feature_lists]

        else:

            train[:] = self.scaler.transform(train)

        return train


class FeatureElimination:

    valid_eliminators = {
        "drop_constant_features": elimination.DropConstantFeatures,
        "drop_low_gini_features": elimination.DropLowGiniFeatures,
        "drop_null_features": elimination.DropNullFeatures,
        "correlated_lower_gini_feature_elimination": elimination.DropCorrelatedLowerGiniFeatures
    }

    valid_params = dict(
        zip(
            valid_eliminators.keys(),
            map(getArgumentsList, valid_eliminators.values())
        )
    )

    def __init__(self, eliminator, params):

        self._validate_eliminator(eliminator, params)

        self.eliminator = eliminator
        self.params = params

        # Check if this class takes estimator as input.
        if "estimator" in inspect.signature(FeatureElimination.valid_eliminators[self.eliminator]).parameters:
            if not isinstance(self.params["estimator"], str):
                pass
            elif self.params["estimator"] == "lightgbm":
                self.params["estimator"] = classifier.LGBMClassifier(n_jobs=1)
            elif self.params["estimator"] == "logistic_regression":
                self.params["estimator"] = classifier.LogisticRegression2(solver='lbfgs', random_state=42)
            else:
                raise ValueError("Invalid string estimator for elimination. Available: `lightgbm` and `logistic_regression`.")

        self.eliminator_instance = FeatureElimination.valid_eliminators[self.eliminator](**self.params)

        self.dropped_columns = []

    @classmethod
    def _validate_eliminator(cls, eliminator, params):
        if eliminator not in cls.valid_eliminators:
            raise Exception(f'{eliminator} is not a valid eliminator.')

        for key in params.keys():
            if key not in cls.valid_params[eliminator]:
                raise Exception(f'{key} is not a valid parameter for {eliminator}.')

    def fit(self, train, y=None, ):
        self.eliminator_instance.fit(train, y)

    def transform(self, X: pd.DataFrame, y=None, sample='Test'):

        logger = getLogger()

        if sample=='Apply' and hasattr(self,'train_features'):

            dropped_df = X
            logger.info("Feature Elimination applied...\n")

        else:

            initial_cols = set(X.columns)
            dropped_df = self.eliminator_instance.transform(X)
            self.dropped_columns = list(initial_cols - set(dropped_df.columns))
            dropped_df = dropped_df[sorted(dropped_df.columns)]

        return dropped_df


class opt_binning:

    def __init__(self, num_variables, cat_variables):
        self.binner = None
        self.num_variables = num_variables
        self.cat_variables = cat_variables

    def fit(self, train, y, param_grid=None):

        self.binner = binning.opt_binning(self.num_variables, self.cat_variables)

        self.binner.fit(train, y, param_grid)

        self.train_features = train.columns

    def transform(self, train, y=None, sample='Test'):

        logger = getLogger()

        if sample == 'Apply' and hasattr(self,'train_features'):

            apply, feature_lists = get_feature_list(self.train_features, train)

            apply = self.binner.transform(apply)

            train[feature_lists] = apply[feature_lists]
            logger.info("Binning applied...\n")

        else:

            train = self.binner.transform(train)

        return train

    def get_binning_table(self):

        self.keys, self.binning_tables, self.gini_table = self.binner.get_binning_table()

        self.binning_tables = pd.concat(self.binning_tables, keys=self.keys)
        if os.getenv('MLFLOW_LOGGING'):
            import mlflow
            mlflow.set_tracking_uri(os.getenv("MLFLOW_URL", 'http://127.0.0.1:5000'))
            self.binning_tables.astype(str).to_excel('binning_tables.xlsx')
            mlflow.log_artifact("binning_tables.xlsx")
            self.gini_table.to_excel('gini_table.xlsx')
            mlflow.log_artifact("gini_table.xlsx")
            os.remove('binning_tables.xlsx')
            os.remove('gini_table.xlsx')

        return self.binning_tables, self.gini_table


class model_LR:

    def __init__(self,
                 param_opt,
                 param_opt_method,
                 train_size,
                 nprocs,
                 random_state,
                 refit_to_all,
                 n_iter,
                 baseEstimator,
                 ############
                 param_grid,
                 cv,
                 verbose,
                 alpha,
                 time_limit,
                 n_feature_params,
                 importance_direction_type,
                 targetcorr_df
                  ):

        self.param_opt=param_opt
        self.param_opt_method=param_opt_method
        self.train_size=train_size
        self.nprocs=nprocs
        self.random_state=random_state
        self.refit_to_all=refit_to_all
        self.n_iter=n_iter
        self.baseEstimator=baseEstimator
        self.param_grid=param_grid
        self.cv = cv
        self.verbose = verbose
        self.alpha = alpha
        self.model=None
        self.time_limit = time_limit
        self.n_feature_params = n_feature_params
        self.importance_direction_type = importance_direction_type
        self.targetcorr_df = targetcorr_df


    def fit(self, train, y, x_val=None, y_val=None, sample='Validation'):

        self.model=classifier.LogisticRegression(param_opt=self.param_opt,
                                                 param_opt_method=self.param_opt_method,
                                                 n_iter = self.n_iter,
                                                 train_size = self.train_size,
                                                 nprocs=self.nprocs,
                                                 refit_to_all = self.refit_to_all,
                                                 random_state=self.random_state,
                                                 baseEstimator=self.baseEstimator,
                                                 param_grid =  self.param_grid,
                                                 cv=self.cv,
                                                 verbose=self.verbose,
                                                 alpha = self.alpha,
                                                 time_limit = self.time_limit,
                                                 n_feature_params = self.n_feature_params,
                                                 importance_direction_type =self.importance_direction_type,
                                                 targetcorr_df = self.targetcorr_df
                                                 )


        self.model.fit_model(train, y)

        if x_val is not None and len(x_val) > 0:
            self.model.predict(x_val, y_val, sample=sample)

    def transform(self, test, y_test=[], sample='Test'):

            dfx_probs, dfx_preds=self.model.predict(test, y_test=y_test, sample='Test')

            return dfx_probs, dfx_preds


class model_RF:

    def __init__(self,
                 param_opt,
                 param_opt_method,
                 train_size,
                 nprocs,
                 random_state,
                 refit_to_all,
                 n_iter,
                 baseEstimator,
                 ############
                 param_grid,
                 cv,
                 verbose,
                 alpha,
                 time_limit,
                 n_feature_params,
                 importance_direction_type,
                 targetcorr_df
                  ):

        self.param_opt=param_opt
        self.param_opt_method=param_opt_method
        self.train_size=train_size
        self.nprocs=nprocs
        self.random_state=random_state
        self.refit_to_all=refit_to_all
        self.n_iter=n_iter
        self.baseEstimator=baseEstimator
        self.param_grid=param_grid
        self.cv = cv
        self.verbose = verbose
        self.alpha = alpha
        self.model=None
        self.time_limit = time_limit
        self.n_feature_params = n_feature_params
        self.importance_direction_type = importance_direction_type
        self.targetcorr_df = targetcorr_df


    def fit(self, train, y, x_val=None, y_val=None, sample='Validation'):
        self.model=classifier.RandomForest(param_opt=self.param_opt,
                                           param_opt_method=self.param_opt_method,
                                           n_iter = self.n_iter,
                                           train_size = self.train_size,
                                           nprocs=self.nprocs,
                                           refit_to_all = self.refit_to_all,
                                           random_state=self.random_state,
                                           baseEstimator=self.baseEstimator,
                                           param_grid =  self.param_grid,
                                           cv=self.cv,
                                           verbose=self.verbose,
                                           alpha = self.alpha,
                                           time_limit = self.time_limit,
                                           n_feature_params = self.n_feature_params,
                                           importance_direction_type =self.importance_direction_type,
                                           targetcorr_df = self.targetcorr_df
                                           )


        self.model.fit_model(train, y)

        if x_val is not None and len(x_val) > 0:
            self.model.predict(x_val, y_val, sample=sample)

    def transform(self, test, y_test=[], sample='Test'):

            dfx_probs, dfx_preds=self.model.predict(test, y_test=y_test, sample='Test')

            return dfx_probs, dfx_preds


class model_LGBM:

    def __init__(self,
                 param_opt,
                 param_opt_method,
                 train_size,
                 nprocs,
                 random_state,
                 refit_to_all,
                 n_iter,
                 baseEstimator,
                 ############
                 param_grid,
                 cv,
                 verbose,
                 alpha,
                 time_limit,
                 n_feature_params,
                 importance_direction_type,
                 targetcorr_df
                  ):

        self.param_opt=param_opt
        self.param_opt_method=param_opt_method
        self.train_size=train_size
        self.nprocs=nprocs
        self.random_state=random_state
        self.refit_to_all=refit_to_all
        self.n_iter=n_iter
        self.baseEstimator=baseEstimator
        self.param_grid=param_grid
        self.cv = cv
        self.verbose = verbose
        self.alpha = alpha
        self.model=None
        self.time_limit = time_limit
        self.n_feature_params = n_feature_params
        self.importance_direction_type = importance_direction_type
        self.targetcorr_df = targetcorr_df

    def fit(self, train, y, x_val=None, y_val=None, sample='Validation'):
        self.model=classifier.LGBM(param_opt=self.param_opt,
                                   param_opt_method=self.param_opt_method,
                                   n_iter = self.n_iter,
                                   train_size = self.train_size,
                                   nprocs=self.nprocs,
                                   refit_to_all = self.refit_to_all,
                                   random_state=self.random_state,
                                   baseEstimator=self.baseEstimator,
                                   param_grid =  self.param_grid,
                                   cv=self.cv,
                                   verbose=self.verbose,
                                    alpha = self.alpha,
                                    time_limit = self.time_limit,
                                    n_feature_params = self.n_feature_params,
                                    importance_direction_type = self.importance_direction_type,
                                    targetcorr_df = self.targetcorr_df
                                   )


        self.model.fit_model(train, y)

        if x_val is not None and len(x_val) > 0:
            self.model.predict(x_val, y_val, sample=sample)

    def transform(self, test, y_test=[], sample='Test'):

            dfx_probs, dfx_preds=self.model.predict(test, y_test=y_test, sample='Test')

            return dfx_probs, dfx_preds


class model_XGBoost:

    def __init__(self,
                 param_opt,
                 param_opt_method,
                 train_size,
                 nprocs,
                 random_state,
                 refit_to_all,
                 n_iter,
                 baseEstimator,
                 ############
                 param_grid,
                 cv,
                 verbose,
                 alpha,
                 time_limit,
                 n_feature_params,
                 importance_direction_type,
                 targetcorr_df
                  ):

        self.param_opt=param_opt
        self.param_opt_method=param_opt_method
        self.train_size=train_size
        self.nprocs=nprocs
        self.random_state=random_state
        self.refit_to_all=refit_to_all
        self.n_iter=n_iter
        self.baseEstimator=baseEstimator
        self.param_grid=param_grid
        self.cv = cv
        self.verbose = verbose
        self.alpha = alpha
        self.model=None
        self.time_limit = time_limit
        self.n_feature_params = n_feature_params
        self.importance_direction_type = importance_direction_type
        self.targetcorr_df = targetcorr_df

    def fit(self, train, y, x_val=None, y_val=None, sample='Validation'):
        self.model=classifier.XGBoost(param_opt=self.param_opt,
                                      param_opt_method=self.param_opt_method,
                                      n_iter = self.n_iter,
                                      train_size = self.train_size,
                                      nprocs=self.nprocs,
                                      refit_to_all = self.refit_to_all,
                                      random_state=self.random_state,
                                      baseEstimator=self.baseEstimator,
                                      param_grid =  self.param_grid,
                                      cv=self.cv,
                                      verbose=self.verbose,
                                      alpha = self.alpha,
                                      time_limit = self.time_limit,
                                      n_feature_params = self.n_feature_params,
                                      importance_direction_type = self.importance_direction_type,
                                      targetcorr_df = self.targetcorr_df
                                      )


        self.model.fit_model(train, y)

        if x_val is not None and len(x_val) > 0:
            self.model.predict(x_val, y_val, sample=sample)

    def transform(self, test, y_test=[], sample='Test'):

            dfx_probs, dfx_preds=self.model.predict(test, y_test=y_test, sample='Test')

            return dfx_probs, dfx_preds
