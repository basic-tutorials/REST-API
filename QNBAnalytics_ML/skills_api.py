# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import json
import pickle
from functools import wraps
from collections import defaultdict
from QNBAnalytics_ML import data, skills_library
from QNBAnalytics_ML.helpers import getLogger

np.random.seed(48)

def variable_injection(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        variable_table = None
        if 'table' in kwargs:
            variable_table = kwargs.pop('table')
        else:
            variable_table = self.variable_table.to_dict(orient='records')

        if variable_table is not None:
            dropped_train = None
            dropped_test = None
            user_target = None

            pre_train_df = pd.DataFrame.from_dict(variable_table)
            use_columns = pre_train_df[pre_train_df['use']]['index'].tolist()
            target = pre_train_df[
                pre_train_df.role == 'target']['index'].tolist()

            if target:
                user_target = target[0]
                self.target = user_target
                self.y_train = self.train[self.target]
                self.train = self.train.drop(columns=[self.target])
                if self.val_apply:
                    self.y_val = self.x_val[self.target]
                    self.x_val = self.x_val.drop(columns=[self.target], axis=1)
            if use_columns:
                if self.target:
                    use_columns = list(set(use_columns) - set([self.target]))
                dropped_train = self.train.drop(columns=use_columns,
                                                errors='ignore')
                self.train = self.train[use_columns]
                if self.val_apply:
                    dropped_test = self.x_val.drop(columns=use_columns,
                                                   errors='ignore')
                    self.x_val = self.x_val[use_columns]
                cats = pre_train_df[
                    pre_train_df['index'].isin(use_columns)].loc[
                        pre_train_df['datatypes'] == "categoric"][
                            'index'].tolist()
                nums = pre_train_df[
                    pre_train_df['index'].isin(use_columns)].loc[
                        pre_train_df['datatypes'] == "numeric"][
                            'index'].tolist()
                if cats:
                    if self.target:
                        cats = list(set(cats) - set([self.target]))
                    self.conversion(
                        user_cats=cats,
                    )
                if nums:
                    if self.target:
                        nums = list(set(nums) - set([self.target]))
                    self.conversion(
                        user_nums=nums,
                    )

            self.cat_variables = pre_train_df[
                pre_train_df['index'].isin(use_columns)].loc[
                    pre_train_df['datatypes'] == "categoric"].set_index(
                        ['index']).index
            if self.target:
                self.cat_variables = self.cat_variables.drop([self.target],
                                                             errors='ignore')
            self.num_variables = pre_train_df[
                pre_train_df['index'].isin(use_columns)].loc[
                    pre_train_df['datatypes'] == "numeric"].set_index(
                        ['index']).index
            if self.target:
                self.num_variables = self.num_variables.drop([self.target],
                                                             errors='ignore')
            self.null_df = pre_train_df[
                pre_train_df['index'].isin(use_columns)].set_index(
                    ['index'])['missing_rate']

        func_out = func(self, *args, **kwargs)
        self.data_explore()

        if variable_table is not None:
            used_columns = self.variable_table[
                self.variable_table['use']]['index'].tolist()
            rest_variables_df = pre_train_df[
                ~pre_train_df['index'].isin(used_columns)]
            rest_variables_df.loc[:,"use"] = False
            self.variable_table = self.variable_table.append(
                rest_variables_df, ignore_index=True)
            if self.target:
                self.variable_table = self.variable_table[
                    self.variable_table['index'] != self.target]
            if dropped_train is not None:
                for col in dropped_train.columns.tolist():
                    self.train[col] = dropped_train[col]

            if self.val_apply:
                if dropped_test is not None:
                    for col in dropped_test.columns.tolist():
                        self.x_val[col] = dropped_test[col]

        return func_out
    return wrapper


class BasePipeline:
    """
    Shared set-up for regressor and classifier pipelines. Consists of skills that can be
    used with specific target types.

    Parameters
    ----------
    train: pd.DataFrame, default=None
        train dataset which the skill will be applied on. train dataset consists of the
        training samples before all preprocessing steps.

    y_train: pd.Series or np.array or list, default=None
        target values of the training dataset.

    val: pd.DataFrame, default=None
        validation dataset which the skill will be applied on. Validation dataset
        consists of the validation samples before all preprocessing steps.

    y_val: pd.Series or np.array or list, default=None
        target values of the training dataset.

    logger: Logger, default=None
        logger instance to write the logs. Default logger of the project is used if
        not specified.

    Methods
    -------
    data_read_sql:
        Read data from specified table or file. EDW_USER and EDW_PASS must be
        defined as environment variables. Name of the target columns is
        specified.

    train_test_split:
        Split the read data as train and test. The order of the rows changes after
        sampling. test_size is  the ratio of the number of test samples.

    drop_cols:
        Remove specified columns. Inplace.

    merge:
        Merge two datasets and save the results in self.merged_df .

    test:
        Evaluate test dataset on models and return the results.

    data_explore:
        Classify variables as categoric or numeric. Normalize feature names
        and calculate missing value rates.

    conversion:
        Conversion of data type of variables.

    null_imputation:
        Fill the missing values in the dataset with specified value and method.

    binning:
        Apply binning to all features. thresh specifies the number
        of samples that the binning will be applied.

    encode_categoricals:
         Encode categoric variables with selected methods. Target Encoding
         is applied if method is not specified.

    scaling:
        Scale the numeric features. Strongly recommended to apply before
        training linear models.

    feature_elimination:
        Eliminate features using specified method. Check feature elimination
        documentation for details [1].

    See Also
    -------
    [1] Feature Elimination Documentation: elimination.py

    """

    def __init__(self, train=None, y_train=None, val=None, y_val=None, logger=None):
        self.cat_variables = None
        self.num_variables = None
        self.null_df = None
        self.train = train
        self.y_train = y_train
        self.x_val = val
        self.y_val = y_val
        self.val_apply = False
        self.logger = getLogger() if logger is None else logger

        if self.x_val is not None and len(self.x_val) > 0:
            self.val_apply = True

        self.df = None
        self.target = None
        self.test_apply = False
        self.data_explorer = None
        self.null_impute = None
        self.sampled = False
        self.sampled_df = None
        self.drop_cols_instance = None

        self.df = None
        self.target = None
        self.sampled_df = None
        self.merged_df = None
        self.test_data = None

        self.model_path = {}
        self.model_comparison = None
        self.eliminated_features = defaultdict(list)

    def data_read_sql(self, username=os.getenv('EDW_USER'),
                      password=os.getenv('EDW_PASS'),
                      query="", test_sql="", target=None, index="", chunksize=25000):
        """
        Read data from specified table or file. EDW_USER and EDW_PASS must be
        defined as environment variables. Name of the target columns is
        specified.

        Parameters
        ----------
        username: str
            user name which is required to connect to the data warehouse

        password: str
            password of the user name  which is required to connect to the data warehouse

        query: str, default=None
            sql query to pull train/valid data set from the data warehouse

        test_sql: str, default=None
            sql query to pull test data set from the data warehouse

        target: str, default=None
            target of train data set which is required to run the model

        index: str, default=None
            id of train data set which is required to run the model

        chunksize: int, default=25000
            chunksize which is the number of rows to be pulled per chunk
        """
        logger = getLogger()
        if not query:
            logger.critical("query parameter is required")
            raise Exception("query parameter is required")
        engine = data.connect_to_sql(
            username, password)
        self.train = data.data_load(engine, sql=query, index=index, chunksize=chunksize)
        if target:
            self.target = target
            self.y_train = self.train[target]
            self.train = self.train.drop(columns=[target])
        object_cols = self.train.columns[self.train.dtypes == 'object']
        for col in object_cols:
            col_series = self.train[col]
            col_series[col_series.notna()] = col_series.astype('str')
            self.train[col] = col_series
        if test_sql:
            engine = data.connect_to_sql(
                username, password)
            self.x_val = data.data_load(engine, sql=test_sql, chunksize=chunksize)
            if target:
                self.y_val = self.x_val[target]
                self.x_val = self.x_val.drop(columns=[target], axis=1)
            object_cols = self.x_val.columns[self.x_val.dtypes == 'object']
            for col in object_cols:
                col_series = self.x_val[col]
                col_series[col_series.notna()] = col_series.astype('str')
                self.x_val[col] = col_series
            self.val_apply = True
        engine.dispose()
        self.data_explore()

    @variable_injection
    def train_test_split(self, test_size=0.33, random_state=1, split_style="random", stratify = None):
        """
        Split the read data as train and test. The order of the rows changes after
        sampling. test_size is  the ratio of the number of test samples.

        Parameters
        ----------
        test_size: int or float, default=0.33
            If float, determines the proportion of the dataset to be used for validation. It should be between 0.0 and 1.0
            If int, determines the number of the dataset to be used for validation

        random_state: int, default=1
            Controls the shuffling applied to the data

        split_style: str, default="random"
            split_style controls how shuffle the dataset while splitting train/validation sample
            if random, shuffling randomly
            if not random, uses a specific column in the dataset

        stratify: array-like, default=None
            If not None, data is split in a stratified fashion, using this as the class labels.

        """
        logger = getLogger()
        try:
            self.train, self.x_val = data.train_test(
                df=self.train.join(self.y_train),
                test_size=test_size,
                random_state=random_state,
                split_style=split_style,  stratify = stratify)

            self.y_train = self.train[self.target]
            self.y_val = self.x_val[self.target]

            self.train = self.train.drop(columns=[self.target])
            self.x_val = self.x_val.drop(columns=[self.target])

            self.val_apply = True
        except Exception as e:
            logger.error(f"Train test splitting failed... {e}", exc_info=True)
            raise e

    @variable_injection
    def drop_cols(self, cols_to_drop=None):
        """
        Remove specified columns. Inplace.

        Parameters
        ----------
        cols_to_drop: list, default=None
            cols_to_drop which is list of the variables (columns) to be dropped from the dataset
        """
        logger = getLogger()
        try:
            if cols_to_drop is None:
                raise Exception("cols_to_drop parameter is required")
            drop_col = skills_library.DropCols(cols_to_drop=cols_to_drop)
            self.train = drop_col.transform(self.train)
            if self.x_val is not None and not self.x_val.empty:
                self.x_val = drop_col.transform(self.x_val)
            self.drop_cols_instance = drop_col
            self.data_explore()
            return drop_col
        except Exception as e:
            logger.error(f"Dropping columns failed... {e}", exc_info=True)
            raise e

    @variable_injection
    def merge(self, how="inner", on=None, overwrite_train=True):
        """
        Merge two datasets and save the results in self.merged_df.

        Parameters
        ----------
        how: str, default=="inner", values: {‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}
            Type of merge

        on: str, default=None
            Column or index level names to join on
        """
        logger = getLogger()
        try:
            if self.target:
                self.train[self.target] = self.y_train
                self.extra_x_train[self.target] = self.extra_y_train
            self.train = data.merge(
                self.train, self.extra_x_train, how, on)
            if self.target:
                self.y_train = self.train[self.target]
                self.train = self.train.drop(columns=[self.target])
            if self.val_apply:
                if self.target:
                    self.x_val[self.target] = self.y_val
                    self.extra_x_test[self.target] = self.extra_y_test
                self.x_val = data.merge(
                    self.x_val, self.extra_x_test, how, on)
                if self.target:
                    self.y_val = self.x_val[self.target]
                    self.x_val = self.x_val.drop(columns=[self.target], axis=1)
        except Exception as e:
            logger.error(f"Merge failed... {e}", exc_info=True)
            raise e

    def test(self, test, y_test=[], sample='Test'):
        """
        Evaluate test dataset on models and return the results.

        Parameters
        ----------
        test: pd.Dataframe
            test dataset which will be scored

        y_test: pd.Dataframe, default=None
            target values of the test dataset.
        """
        logger = getLogger()
        try:
            self.test_data = {}
            for key, value in self.model_path.items():
                if "Ensemble" in key:
                    dfx_probs, dfx_preds = value.transform(test, y_test)
                    self.test_data.update({key: [dfx_probs, dfx_preds]})
                elif "Model" in key:
                    dfx_probs, dfx_preds = value.transform(test, y_test)
                    self.test_data.update({key: [dfx_probs, dfx_preds]})
                else:
                    test = value.transform(test, y_test, sample=sample)

            return self.test_data
        except Exception as e:
            logger.error(f"Testing failed... {e}", exc_info=True)
            raise e

    def transform(self, X, y=None):
        logger = getLogger()
        logger.info("Transforming started...")
        for key, value in self.model_path.items():
            if ("Model" not in key) and ("Ensemble" not in key):
                X = value.transform(X, y)

        return X

    def data_explore(self):
        """
        Classify variables as categoric or numeric. Normalize feature names
        and calculate missing value rates.
        """
        logger = getLogger()
        try:
            self.data_explorer = skills_library.data_explore()
            self.data_explorer.fit(self.train, self.y_train)
            self.train = self.data_explorer.transform(self.train)

            if self.val_apply:
                self.x_val = self.data_explorer.transform(self.x_val)
            self.model_path.update({str(len(self.model_path.keys())) + '.Data Explore': self.data_explorer})
            self.variable_table = self.data_explorer.variable_type.reset_index()
            self.cat_variables=self.data_explorer.cat_variables
            self.num_variables=self.data_explorer.num_variables
            self.null_df = self.data_explorer.null_df
            if not hasattr(self, "targetcorr_df"):
                self.targetcorr_df = self.data_explorer.targetcorr_df
            else:
                pass

            return self.data_explorer
        except Exception as e:
            logger.error(f"Data exploration failed... {e}", exc_info=True)
            raise e

    def conversion(self, user_cats=None, user_nums=None):
        """
        Conversion of data type of variables.

        Parameters
        ----------
        user_cats: list, default=None
            numeric variables to be converted to categoryx

        user_nums: list, default=None
            numeric variables to be converted to category
        """
        logger = getLogger()
        try:
            self.converter = skills_library.conversion()
            self.num_variables, self.cat_variables = self.converter.fit(self.num_variables, self.cat_variables, user_cats=user_cats, user_nums=user_nums)
            self.train = self.converter.transform(self.train)

            if self.val_apply:
                self.x_val = self.converter.transform(self.x_val)

            return self.converter

        except Exception as e:
            logger.error(f"Data Type Conversion failed... {e}", exc_info=True)
            raise e

    @variable_injection
    def variable_transformation(self, transformer=None, cols_to_transform="all", params=None):
        logger = getLogger()
        try:
            if params is None:
                params = {}
            self.transformer_instance = skills_library.FeatureTransformation(transformer, cols_to_transform, params)
            logger.info(f"Feature Transformation Started: {transformer}")

            self.train = self.transformer_instance.transform(self.train)

            if self.val_apply:
                self.x_val = self.transformer_instance.transform(self.x_val)

            path_key = str(len(self.model_path.keys())) + f'.{" ".join(map(str.capitalize, transformer.split("_")))}'
            self.model_path.update({path_key: self.transformer_instance})
            logger.info("Feature Transformation completed.")

            return self.transformer_instance
        except Exception as e:
            logger.error(f"Feature Transformation failed... {e}", exc_info=True)
            raise e

    @variable_injection
    def null_imputation(self, null_thresh=99.99999999, fill_value='Null', num_fill_value=0, num_null=True, cat_null=True,
                        num_strategy='mean', cat_strategy='constant'):
        """
        Fill the missing values in the dataset with specified value and method.

        Parameters
        ----------
        null_thresh: float, default=99.9
            if the share of null values in a column is greater than null_thresh, the column is be droped
            It should be between 0.0 and 1.0

        fill_value: str, default='Null'
            fill_value which is used to impute null values in categorical variables

        num_fill_value: list, default=None
            num_fill_value which is used to impute null values in numeric variables

        num_null: bool, default=True
            if True, applies null_imputation method to numeric variables

        cat_null: bool, default=True
            if True, applies null_imputation method to numeric variables

        num_strategy: str, default='mean', values: {‘mean’}
            The imputation strategy for numeric variables

        cat_strategy: list, default='constant', values: {‘constant’}
            The imputation strategy for categorical variables
        """
        logger = getLogger()
        try:
            self.null_impute = skills_library.null_imputation(null_thresh=null_thresh,
                                                              fill_value=fill_value,
                                                              num_fill_value=num_fill_value,
                                                              num_null=num_null,
                                                              cat_null=cat_null,
                                                              num_strategy=num_strategy,
                                                              cat_strategy=cat_strategy
                                                              )
            self.null_impute.fit(self.train, self.data_explorer.null_df, self.data_explorer.num_variables,
                                 self.data_explorer.cat_variables)
            self.train = self.null_impute.transform(self.train)
            if self.val_apply:
                self.x_val = self.null_impute.transform(self.x_val)
            self.model_path.update({str(len(self.model_path.keys())) + '.Null Imputation': self.null_impute})

            return self.null_impute
        except Exception as e:
            logger.error(f"Null imputation failed... {e}", exc_info=True)
            raise e

    @variable_injection
    def binning(self, method="optbinning", thresh=100000, nprocs=5, max_n_bins=30, min_bin_size=0.0001, monotonic_trend='auto_asc_desc', param_grid=None, use_existing=False):
        """
        Apply binning to all features. thresh specifies the number
        of samples that the binning will be applied.

        Parameters
        ----------
        method: str, default="optbinning"
            The binning method

        thresh: int, default=100000
            It determines the number of the dataset which the binning method is applied on
            The parameter is not used for optbinning

        nprocs: int, default=5
            The number of jobs to run in parallel

        max_n_bins: int, default=30
            The maximum number of bins

        min_bin_size: bool, default=0.0001
            The fraction of minimum number of rows for each bin

        monotonic_trend*: str, default='auto_asc_desc'
            The event rate monotonic trend

        param_grid*: list, default=None
            Parameter grids for optbinning method

        *more details for optbinning method: http://gnpalencia.org/optbinning/binning_binary.html
        """
        logger = getLogger()

        if method == "optbinning":

            if type(param_grid) == str:
                param_grid = json.loads(param_grid)
            if param_grid is None:
                param_grid = {
                                'max_n_prebins': 100,
                                'max_n_bins': max_n_bins,
                                'min_bin_size': min_bin_size,
                                'monotonic_trend': monotonic_trend,
                                'min_event_rate_diff': 0,
                                'user_splits': None,
                                'user_splits_fixed': None
                            }

            else:
                param_grid2 = {
                    'max_n_bins': max_n_bins,
                    'min_bin_size': min_bin_size,
                    'monotonic_trend': monotonic_trend
                }

                ea_list = ['max_n_bins','min_bin_size','monotonic_trend']
                for name in ea_list:
                    if name in param_grid:
                        param_grid2.pop(name, None)


                param_grid = {**param_grid, **param_grid2}

            try:
                logger.info("Opt binning started... ")
                
                if use_existing==False:
                    self.monotonic_binning = skills_library.opt_binning(
                        num_variables=self.num_variables, cat_variables=self.cat_variables)
                    self.monotonic_binning.fit(self.train, self.y_train, param_grid=param_grid)

                elif type(use_existing) == str:
                    with open(use_existing, 'rb') as f:
                        existing_binning = pickle.load(f)
                    self.monotonic_binning = existing_binning
                
                self.train = self.monotonic_binning.transform(self.train)
                if self.val_apply:
                    self.x_val = self.monotonic_binning.transform(self.x_val)

                self.model_path.update({str(len(self.model_path.keys())) + '.Binning': self.monotonic_binning})
                self.monotonic_binning.get_binning_table()
                logger.info("Opt binning completed... ")
                return self.monotonic_binning
            except Exception as e:
                logger.error(f"Opt binning failed... {e}", exc_info=True)
                raise e



    @variable_injection
    def encode_categoricals(self, cat_fix=None, method="Target Encoder", sample=200000, min_samples_leaf=50):
        """
        Encode categoric variables with selected methods. Target Encoding
        is applied if method is not specified.

        Parameters
        ----------
        cat_fix: list, default=None
            It determines categorical variables which the encoding method is applied on
            If None, uses all categorical variables that are determined during null imputation method

        method: str, default="Target Encoder", values: {"Target Encoder"}
            The encoding method

        sample: int, default=200000
            It determines the number of the dataset which the encoding method is applied on

        min_samples_leaf: int, default=50
            The minimum number of leafs
        """
        if cat_fix is None:
            cat_fix = []
        logger = getLogger()
        try:
            if len(cat_fix) == 0:
                cat_fix = self.cat_variables
            self.encode_categorical = skills_library.encode_categoricals(cat_fix=cat_fix,
                                                                    method=method,
                                                                    sample=sample,
                                                                    min_samples_leaf=min_samples_leaf
                                                                    )
            self.encode_categorical.fit(self.train, self.y_train)
            self.train = self.encode_categorical.transform(self.train)
            if self.val_apply:
                self.x_val = self.encode_categorical.transform(self.x_val)
            self.model_path.update({str(len(self.model_path.keys())) + '.Encoding': self.encode_categorical})

            return self.encode_categorical
        except Exception as e:
            logger.error(f"Encoding categoricals failed... {e}", exc_info=True)
            raise e

    @variable_injection
    def scaling(self, type='StandardScaler', copy=True, with_mean=True, with_std=True, feature_range=(0, 1)):
        """
        Scale the numeric features. Strongly recommended to apply before
        training linear models.

        Parameters
        ----------
        type: str, default=”StandardScaler”, values: {"StandardScaler"}
            Scaling method

        copy: bool, default=True
            If True, copy the dataset

        with_mean: bool, default=True
            If True, center the data before scaling

        with_std: bool, default=True
            If True, scale the data to unit variance (or standart deviation)

        feature_range: tuple (min, max), default=(0,1)
            Desired range of transformed data(Only MinMaxScaler method uses this parameter)
        """
        logger = getLogger()
        try:
            self.scale = skills_library.scaling(type=type, copy=copy, with_mean=with_mean, with_std=with_std, feature_range=feature_range)
            self.scale.fit(self.train)
            self.train = self.scale.transform(self.train)
            if self.val_apply:
                self.x_val = self.scale.transform(self.x_val)
            self.model_path.update({str(len(self.model_path.keys())) + '.Scaling': self.scale})

            return self.scale
        except Exception as e:
            logger.error(f"Scaling failed... {e}", exc_info=True)
            raise e

    @variable_injection
    def feature_elimination(self, eliminator=None, params=None):
        """
        Select features using univariate/multivariate and model based methods.
        
        Parameters
        ----------
        eliminator: str, default=None
        params: dict, default=None

        Available Elimination methods
        -----------------------------
        * drop_constant_features
            DropConstantFeatures() drops constant and quasi-constant variables from a dataframe.
             Constant variables show the same value in all the observations in the dataset.
             Quasi-constant variables show the same value in almost all the observations in the dataset.
            - Parameters: ['variables', 'tol', 'missing_values', 'confirm_variables']

        * drop_low_gini_features
            DropLowGiniFeatures removes features with gini value lower than given threshold.
            - Parameters: ['variables', 'threshold', 'missing_values', 'confirm_variables']

        * drop_null_features
            DropNullFeatures removes features with null values higher than threshold.
            - Parameters: ['variables', 'threshold', 'confirm_variables']

        * correlated_lower_gini_feature_elimination
            DropCorrelatedLowGiniFeatures finds and removes correlated features similar to DropCorrelatedFeatures,
            however instead of first round first removed, it removes the feature with lower gini values.
            - Parameters: ['variables', 'threshold', 'missing_values', 'confirm_variables']
    
        * Check feature elimination documentation for parameter detail 'QNBAnalyticsML.elimination'.
        """


        logger = getLogger()
        try:
            if params is None:
                params = {}
            eliminator_instance = skills_library.FeatureElimination(eliminator, params)
            logger.info(f"Feature Elimination Started: {eliminator}")
            eliminator_instance.fit(self.train, self.y_train)
            self.train = eliminator_instance.transform(self.train)

            if self.val_apply:
                self.x_val = eliminator_instance.transform(self.x_val)

            self.eliminated_features[eliminator] += eliminator_instance.dropped_columns
            path_key = str(len(self.model_path.keys())) + f'.{" ".join(map(str.capitalize, eliminator.split("_")))}'
            self.model_path.update({path_key: eliminator_instance})
            logger.info("Feature Elimination completed.")

            return eliminator_instance
        except Exception as e:
            logger.error(f"Feature Elimination failed... {e}", exc_info=True)
            raise e




class ClassifierPipeline(BasePipeline):
    """
    Pipeline class for classification tasks.

    Parameters
    ----------

    Methods
    ----------
    LogisticRegression:
        Logistic Regression classification model. Uses LogisticRegression class
        of scikit-learn library with parameter optimization steps.

    RandomForest:
         Random Forest classification model. Uses RandomForestClassifier of
         scikit-learn library with parameter optimization steps.

    LGBM:
         LGBM classification model. Uses LGBMClassifier of LightGBM library with
         parameter optimization steps.

    XGBoost:
         XGBoost classification model. Uses XGBClassifier of XGBoost library with
         parameter optimization steps.

    """

    def __init__(self, train=None, y_train=None, val=None, y_val=None, logger=None):
        super().__init__(train, y_train, val, y_val, logger)

        self.logistic_regression_model = None
        self.decision_tree_model = None
        self.random_forest_model = None
        self.lgbm_model = None
        self.xgboost_model = None
        self.mlp_model = None

    @variable_injection
    def LogisticRegression(self,
                           param_opt=True,
                           param_opt_method="RandomizedSearchCV",
                           train_size=None,
                           nprocs=1,
                           random_state=48,
                           refit_to_all=True,
                           n_iter=15,
                           base_estimator='GP',
                           ############
                           param_grid=None,
                           cv=3,
                           verbose=1,
                           alpha=None,
                           time_limit=None,
                           n_feature_params=15,
                           importance_direction_type="shap"
                           ):
        """
        Logistic Regression classification model. Uses LogisticRegression class
        of scikit-learn library with parameter optimization steps.

        Parameters
        ----------
        param_opt: bool, default=True
            If True, uses parameter opzimization method

        param_opt_method: str, default="RandomizedSearchCV", values: {"RandomizedSearchCV"}
            The parameter optimization method

        train_size: int, default=None
            Determines the number of the dataset which the model is applied on

        nprocs: bool, default=1
            If True, scale the data to unit variance (or standart deviation)

        refit_to_all: bool, default=True
            If True, the model is applied on all of the dataset

        n_iter: int, default=15
            The number of parameter sets to be used for the parameter optimization

        base_estimator: str, default='GP', values: {"GP"}
            Base estimator of the parameter optimization method.
            Used for BayesSearchCV
            GP: Gaussian Process

        param_grid*: dict, default=None
            The parameter grid to run the model

        cv: int, default=3
            If True, scale the data to unit variance (or standart deviation)

        verbose: int, default=1
            If True, scale the data to unit variance (or standart deviation)

        alpha: float, default=None
            Model selection parameter which is used in the Parameter Optimization Method
            Formula for selection: validation gini + (train gini - validation gibi) * alpha

        time_limit: int, default=None

        n_feature_params: int, default=15
            The number of features which are shown in the model feature plot

        importance_direction_type: str, default="shap"
            Specify technique to select feature's positive/negative correlation to the target.
            Feature importances and target correlationscan be found in feature importances dataframe.

            For importance_direction_type='shap'
                * Pearson correlation between processed features and shap values are calculated.
                Processed features are train data after preprocessing steps applieds by the user.

        *more details: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        """

        if type(param_grid) == str:
            param_grid = json.loads(param_grid)

        logger = getLogger()
        try:
            self.logistic_regression_model = skills_library.model_LR(param_opt=param_opt,
                                                                     param_opt_method=param_opt_method,
                                                                     n_iter=n_iter,
                                                                     train_size=train_size,
                                                                     nprocs=nprocs,
                                                                     refit_to_all=refit_to_all,
                                                                     random_state=random_state,
                                                                     baseEstimator=base_estimator,
                                                                     param_grid=param_grid,
                                                                     cv=cv,
                                                                     verbose=verbose,
                                                                     alpha=alpha,
                                                                     time_limit=time_limit,
                                                                     n_feature_params=n_feature_params,
                                                                     importance_direction_type=importance_direction_type,
                                                                     targetcorr_df = self.targetcorr_df
                                                                     )
            self.logistic_regression_model.fit(self.train, self.y_train, self.x_val, self.y_val)

            path_key = str(len(self.model_path.keys())) + '.Model: Logistic Regression'
            self.model_path.update({path_key: self.logistic_regression_model})

            return self.logistic_regression_model
        except Exception as e:
            logger.error(f"Running Linear Regression failed... {e}", exc_info=True)
            raise e


    @variable_injection
    def RandomForest(self,
                     param_opt=True,
                     param_opt_method="RandomizedSearchCV",
                     train_size=None,
                     nprocs=1,
                     random_state=48,
                     refit_to_all=True,
                     n_iter=10,
                     base_estimator='GP',
                     ############
                     param_grid=None,
                     cv=3,
                     verbose=1,
                     alpha=None,
                     time_limit=None,
                     n_feature_params=15,
                     importance_direction_type="shap"
                     ):
        """
        Random Forest classification model. Uses RandomForestClassifier of
        scikit-learn library with parameter optimization steps.

        Parameters
        ----------
        param_opt: bool, default=True
            If True, uses parameter opzimization method

        param_opt_method: str, default="RandomizedSearchCV", values: {"RandomizedSearchCV"}
            The parameter optimization method

        train_size: int, default=None
            Determines the number of the dataset which the model is applied on

        nprocs: bool, default=1
            If True, scale the data to unit variance (or standart deviation)

        refit_to_all: bool, default=True
            If True, the model is applied on all of the dataset

        n_iter: int, default=15
            The number of parameter sets to be used for the parameter optimization

        base_estimator: str, default='GP', values: {"GP", "RF", "ET", "GBRT"}
            Base estimator of the parameter optimization method.
            Used for BayesSearchCV
            GP: Gaussian Process, RF: Random Forest, ET: Extra Trees Regressor, GBRT: Gradient Boosting Regression Trees

        param_grid*: dict, default=None
            The parameter grid to run the model

        cv: int, default=3
            If True, scale the data to unit variance (or standart deviation)

        verbose: int, default=1
            If True, scale the data to unit variance (or standart deviation)

        alpha: float, default=None
            Model selection parameter which is used in the Parameter Optimization Method
            Formula for selection: validation gini + (train gini - validation gibi) * alpha

        time_limit: int, default=None

        n_feature_params: int, default=15
            The number of features which are shown in the model feature plot

        importance_direction_type: str, default="shap"
            Specify technique to select feature's positive/negative correlation to the target.
            Feature importances and target correlationscan be found in feature importances dataframe.

            For importance_direction_type='factor'
                * Pearson correlations between raw features and target are calculated. 
                Raw features are train data given by user without anypreprocessing.
            For importance_direction_type='shap'
                * Pearson correlation between processed features and shap values are calculated.
                Processed features are train data after preprocessing steps applieds by the user.

        *more details: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        """
        if type(param_grid) == str:
            param_grid = json.loads(param_grid)

        logger = getLogger()
        try:
            self.random_forest_model = skills_library.model_RF(param_opt=param_opt,
                                                               param_opt_method=param_opt_method,
                                                               n_iter=n_iter,
                                                               train_size=train_size,
                                                               nprocs=nprocs,
                                                               refit_to_all=refit_to_all,
                                                               random_state=random_state,
                                                               baseEstimator=base_estimator,
                                                               param_grid=param_grid,
                                                               cv=cv,
                                                               verbose=verbose,
                                                               alpha=alpha,
                                                               time_limit=time_limit,
                                                               n_feature_params=n_feature_params,
                                                               importance_direction_type=importance_direction_type,
                                                               targetcorr_df = self.targetcorr_df
                                                               )
            self.random_forest_model.fit(self.train, self.y_train, self.x_val, self.y_val)

            path_key = str(len(self.model_path.keys())) + '.Model: Random Forest'
            self.model_path.update({path_key: self.random_forest_model})

            return self.random_forest_model
        except Exception as e:
            logger.error(f"Running Random Forest failed... {e}", exc_info=True)
            raise e

    @variable_injection
    def LGBM(self,
             param_opt=True,
             param_opt_method="RandomizedSearchCV",
             n_iter=30,
             train_size=None,
             refit_to_all=True,
             nprocs=1,
             random_state=48,
             base_estimator='GP',
             param_grid=None,
             cv=3,
             verbose=1,
             alpha=None,
             time_limit=None,
             n_feature_params=15,
             importance_direction_type="shap"
             ):
        """
        LGBM classification model. Uses LGBMClassifier of LightGBM library with
        parameter optimization steps.

        Parameters
        ----------
        param_opt: bool, default=True
            If True, uses parameter opzimization method

        param_opt_method: str, default="RandomizedSearchCV", values: {"RandomizedSearchCV"}
            The parameter optimization method

        train_size: int, default=None
            Determines the number of the dataset which the model is applied on

        nprocs: bool, default=1
            If True, scale the data to unit variance (or standart deviation)

        refit_to_all: bool, default=True
            If True, the model is applied on all of the dataset

        n_iter: int, default=15
            The number of parameter sets to be used for the parameter optimization

        base_estimator: str, default='GP', values: {"GP", "RF", "ET", "GBRT"}
            Base estimator of the parameter optimization method.
            Used for BayesSearchCV
            GP: Gaussian Process, RF: Random Forest, ET: Extra Trees Regressor, GBRT: Gradient Boosting Regression Trees

        param_grid*: dict, default=None
            The parameter grid to run the model

        cv: int, default=3
            If True, scale the data to unit variance (or standart deviation)

        verbose: int, default=1
            If True, scale the data to unit variance (or standart deviation)

        alpha: float, default=None
            Model selection parameter which is used in the Parameter Optimization Method
            Formula for selection: validation gini + (train gini - validation gibi) * alpha

        time_limit: int, default=None

        n_feature_params: int, default=15
            The number of features which are shown in the model feature plot

        importance_direction_type: str, default="shap"
            Specify technique to select feature's positive/negative correlation to the target.
            Feature importances and target correlationscan be found in feature importances dataframe.

            For importance_direction_type='factor'
                * Pearson correlations between raw features and target are calculated. 
                Raw features are train data given by user without anypreprocessing.
            For importance_direction_type='shap'
                * Pearson correlation between processed features and shap values are calculated.
                Processed features are train data after preprocessing steps applieds by the user.

        *more details: https://lightgbm.readthedocs.io/en/latest/Parameters.html
        """
        if type(param_grid) == str:
            param_grid = json.loads(param_grid)

        logger = getLogger()
        try:
            self.lgbm_model = skills_library.model_LGBM(param_opt=param_opt,
                                                        param_opt_method=param_opt_method,
                                                        n_iter=n_iter,
                                                        train_size=train_size,
                                                        nprocs=nprocs,
                                                        refit_to_all=refit_to_all,
                                                        random_state=random_state,
                                                        baseEstimator=base_estimator,
                                                        param_grid=param_grid,
                                                        cv=cv,
                                                        verbose=verbose,
                                                        alpha=alpha,
                                                        time_limit=time_limit,
                                                        n_feature_params=n_feature_params,
                                                        importance_direction_type=importance_direction_type,
                                                        targetcorr_df = self.targetcorr_df
                                                        )
            self.lgbm_model.fit(self.train, self.y_train, self.x_val, self.y_val)

            path_key = str(len(self.model_path.keys())) + '.Model: LGBM'
            self.model_path.update({path_key: self.lgbm_model})

            return self.lgbm_model
        except Exception as e:
            logger.error(f"Running LGBM failed... {e}", exc_info=True)
            raise e

    @variable_injection
    def XGBoost(self,
                param_opt=True,
                param_opt_method="RandomizedSearchCV",
                n_iter=15,
                train_size=None,
                refit_to_all=False,
                nprocs=1,
                random_state=48,
                base_estimator='GP',
                param_grid=None,
                cv=3,
                verbose=1,
                alpha=None,
                time_limit=None,
                n_feature_params=15,
                importance_direction_type="shap"
                ):
        """
        XGBoost classification model. Uses XGBClassifier of XGBoost library with
        parameter optimization steps.

        Parameters
        ----------
        param_opt: bool, default=True
            If True, uses parameter opzimization method

        param_opt_method: str, default="RandomizedSearchCV", values: {"RandomizedSearchCV"}
            The parameter optimization method

        train_size: int, default=None
            Determines the number of the dataset which the model is applied on

        nprocs: bool, default=1
            If True, scale the data to unit variance (or standart deviation)

        refit_to_all: bool, default=True
            If True, the model is applied on all of the dataset

        n_iter: int, default=15
            The number of parameter sets to be used for the parameter optimization

        base_estimator: str, default='GP', values: {"GP", "RF", "ET", "GBRT"}
            Base estimator of the parameter optimization method.
            Used for BayesSearchCV
            GP: Gaussian Process, RF: Random Forest, ET: Extra Trees Regressor, GBRT: Gradient Boosting Regression Trees

        param_grid*: dict, default=None
            The parameter grid to run the model

        cv: int, default=3
            If True, scale the data to unit variance (or standart deviation)

        verbose: int, default=1
            If True, scale the data to unit variance (or standart deviation)

        alpha: float, default=None
            Model selection parameter which is used in the Parameter Optimization Method
            Formula for selection: validation gini + (train gini - validation gibi) * alpha

        time_limit: int, default=None

        n_feature_params: int, default=15
            The number of features which are shown in the model feature plot

        importance_direction_type: str, default="shap"
            Specify technique to select feature's positive/negative correlation to the target.
            Feature importances and target correlationscan be found in feature importances dataframe.

            For importance_direction_type='factor'
                * Pearson correlations between raw features and target are calculated. 
                Raw features are train data given by user without anypreprocessing.
            For importance_direction_type='shap'
                * Pearson correlation between processed features and shap values are calculated.
                Processed features are train data after preprocessing steps applieds by the user.

        *more details: https://xgboost.readthedocs.io/en/stable/parameter.html
        """
        if type(param_grid) == str:
            param_grid = json.loads(param_grid)

        logger = getLogger()
        try:      
            self.xgboost_model = skills_library.model_XGBoost(param_opt=param_opt,
                                                              param_opt_method=param_opt_method,
                                                              n_iter=n_iter,
                                                              train_size=train_size,
                                                              nprocs=nprocs,
                                                              refit_to_all=refit_to_all,
                                                              random_state=random_state,
                                                              baseEstimator=base_estimator,
                                                              param_grid=param_grid,
                                                              cv=cv,
                                                              verbose=verbose,
                                                              alpha=alpha,
                                                              time_limit=time_limit,
                                                              n_feature_params=n_feature_params,
                                                              importance_direction_type=importance_direction_type,
                                                              targetcorr_df = self.targetcorr_df
                                                              )
            self.xgboost_model.fit(self.train, self.y_train, self.x_val, self.y_val)

            path_key = str(len(self.model_path.keys())) + '.Model: XGBoost'
            self.model_path.update({path_key: self.xgboost_model})

            return self.xgboost_model
        except Exception as e:
            logger.error(f"Running XGBoost failed... {e}", exc_info=True)
            raise e

