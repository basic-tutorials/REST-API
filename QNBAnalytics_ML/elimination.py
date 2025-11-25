import pandas as pd
import logging

from feature_engine.selection.base_selector import BaseSelector
from feature_engine.selection import DropConstantFeatures
from feature_engine.dataframe_checks import check_X


try:
    # For feature_engine >= 1.4
    from feature_engine.variable_handling import (
        check_numerical_variables,
        find_all_variables,
        find_numerical_variables,
        retain_variables_if_in_df,
    )
    _find_or_check_numerical_variables = check_numerical_variables
    _find_all_variables = find_all_variables
    _find_numerical_variables = find_numerical_variables
    _retain_variables_if_in_df = retain_variables_if_in_df
    _check_input_parameter_variables = lambda x: None  # Not needed in newer versions
except ImportError:
    # For feature_engine < 1.4
    from feature_engine.variable_manipulation import (
        _check_input_parameter_variables,
        _find_or_check_numerical_variables,
        _find_all_variables,
    )
    _retain_variables_if_in_df = None  # Not available in older versions

from feature_engine.tags import _return_tags

from sklearn.metrics import roc_auc_score
from sklearn.utils.validation import check_is_fitted

from time import perf_counter
from contextlib import contextmanager


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
logger.addHandler(console)


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start



__all__ = [
    "DropNullFeatures",
    "DropConstantFeatures",
    "DropCorrelatedLowerGiniFeatures",
    "DropLowGiniFeatures",
]


class DropConstantFeatures(DropConstantFeatures):
    def transform(self, X: pd.DataFrame, y=None):
        """
        Return dataframe with selected features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The input dataframe.

        y: Not used.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_selected_features]
            Pandas dataframe with the selected features.
        """

        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = check_X(X)

        # get selected features.
        self.selected_features = list(set(self.variables_) - set(self.features_to_drop_))

        for col in self.selected_features:
            if col not in X.columns:
                raise ValueError(f"Selected column: {col} not found in the input dataframe.")

        return X[self.selected_features]


class DropLowGiniFeatures(BaseSelector):
    """
    Drop low gini features given a threshold. Only works on numerical variables and binary and multiclass classification tasks.

    Parameters
    ----------
    variables: str or list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate
        all numerical features in the dataset.

    threshold: float, default=0.01
        The gini threshold below which a feature will be deemed to have too little predictive power and dropped from dataset.

    missing_values: str, default=ignore
        Whether the missing values should be raised as error or ignored when
        determining correlation. Takes values 'raise' and 'ignore'.

    confirm_variables: bool, default=False
            If set to True, variables that are not present in the input dataframe will
            be removed from the list of variables. Only used when passing a variable
            list to the parameter `variables`. See parameter variables for more details.

    Attributes
    ----------
    features_to_drop_:
        List with the features that will be removed.

    gini_table:
        Dictionary of {variable name: gini value}.

    variables_:
        The variables that will be considered for the feature selection procedure.

    feature_names_in_:
        List with the names of features seen during `fit`.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Find correlated features.

    fit_transform:
        Fit to data, then transform it.

    get_feature_names_out:
        Get output feature names for transformation.

    get_params:
        Get parameters for this estimator.

    set_params:
        Set the parameters of this estimator.

    transform:
        Remove correlated features.

    Notes
    --------
    Gini value makes a assumption that relationship of a variable to the target is an always increasing/decreasing function.
    For some datasets low gini features might have strong predictive power, as a results use it conservatively.

    """

    def __init__(
        self,
        variables=None,
        threshold=0.01,
        missing_values="raise",
        confirm_variables=False,
    ):
        super().__init__(confirm_variables)

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values can only take ['raise','ignore']")

        self.threshold = threshold
        self.variables = _check_input_parameter_variables(variables)
        self.missing_values = missing_values

    def fit(self, X, y):
        # check input dataframe
        X = check_X(X)

        # check y
        n_class = y.nunique()
        if not (2 <= n_class <= 1000):
            raise ValueError(f"Too many classes ({n_class}). Please check y.")

        # find all numerical variables or check those entered are in the dataframe
        if self.variables is None:
            # Get all numerical variables by default
            self.variables_ = _find_numerical_variables(X)
        else:
            # Get user-specified variables
            if self.confirm_variables and _retain_variables_if_in_df is not None:
                variables = _retain_variables_if_in_df(X, self.variables)
            else:
                variables = self.variables

            # Filter to only numerical variables (intersect with numerical columns)
            numerical_vars = _find_numerical_variables(X)
            self.variables_ = [v for v in variables if v in numerical_vars]

        self.features_to_drop_ = set()
        self.gini_table = {}

        for var in self.variables_:
            if self.missing_values == "ignore":
                X_var_nonull = X[var].dropna()
                _auc = roc_auc_score(
                    y[y.index.isin(X_var_nonull.index)],
                    X_var_nonull,
                    multi_class="ovr",
                    average="weighted",
                )
            else:
                _auc = roc_auc_score(y, X[var], multi_class="ovr", average="weighted")

            gini = abs(2 * _auc - 1)
            self.gini_table[var] = gini

            if gini < self.threshold:
                self.features_to_drop_.add(var)

        # sort gini values
        self.gini_table = dict(
            sorted(self.gini_table.items(), reverse=True, key=lambda item: item[1])
        )

        # save input features
        self._get_feature_names_in(X)

        return self

    def transform(self, X: pd.DataFrame, y=None):
        """
        Return dataframe with selected features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The input dataframe.

        y: Not used.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_selected_features]
            Pandas dataframe with the selected features.
        """

        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = check_X(X)

        # get selected features.
        self.selected_features = list(set(self.variables_) - set(self.features_to_drop_))

        for col in self.selected_features:
            if col not in X.columns:
                raise ValueError(f"Selected column: {col} not found in the input dataframe.")

        return X[self.selected_features]


    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        tags_dict["requires_y"] = True
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        return tags_dict


class DropNullFeatures(BaseSelector):
    """
    Drop highly null features above a threshold.

    Parameters
    ----------
    variables: str or list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate
        all numerical features in the dataset.

    threshold: float, default=1
        The null threshold higher which a feature will be deemed to be uninformative and dropped from dataset.

    confirm_variables: bool, default=False
            If set to True, variables that are not present in the input dataframe will
            be removed from the list of variables. Only used when passing a variable
            list to the parameter `variables`. See parameter variables for more details.

    Attributes
    ----------
    features_to_drop_:
        List with the features that will be removed.

    null_table:
        Dictionary of {variable name: null percetange}.

    variables_:
        The variables that will be considered for the feature selection procedure.

    feature_names_in_:
        List with the names of features seen during `fit`.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Find highly null features.

    fit_transform:
        Fit to data, then transform it.

    get_feature_names_out:
        Get output feature names for transformation.

    get_params:
        Get parameters for this estimator.

    set_params:
        Set the parameters of this estimator.

    transform:
        Remove highly null features.

    Notes
    --------
    -

    """

    def __init__(self, variables=None, threshold=1, confirm_variables=False):
        super().__init__(confirm_variables)
        self.threshold = threshold
        self.variables = _check_input_parameter_variables(variables)

        if not (0 <= threshold <= 1):
            raise ValueError("Threshold should be float and between 0 and 1.")

    def fit(self, X, y=None):
        # check input dataframe
        X = check_X(X)

        # find all variables or check those entered are present in the dataframe
        if self.variables is None:
            self.variables_ = _find_all_variables(X)
        elif self.confirm_variables and _retain_variables_if_in_df is not None:
            self.variables_ = _retain_variables_if_in_df(X, self.variables)
        else:
            self.variables_ = self.variables

        # save input features
        self._get_feature_names_in(X)

        # find null features
        threshold = 1.0 - self.threshold
        selected_features = X.dropna(thresh=len(X) * threshold, axis=1).columns

        # create null_table
        self.null_table = (
            (X.isnull().sum() / len(X)).sort_values(ascending=False).to_dict()
        )

        # feature to drop
        self.features_to_drop_ = set(self.variables_) - set(selected_features)

        return self

    def transform(self, X: pd.DataFrame, y=None):
        """
        Return dataframe with selected features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The input dataframe.

        y: Not used.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_selected_features]
            Pandas dataframe with the selected features.
        """

        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = check_X(X)

        # get selected features.
        self.selected_features = list(set(self.variables_) - set(self.features_to_drop_))

        for col in self.selected_features:
            if col not in X.columns:
                raise ValueError(f"Selected column: {col} not found in the input dataframe.")

        return X[self.selected_features]

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "all"
        tags_dict["requires_y"] = False
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        return tags_dict



def _calculate_gini_table(X, y, missing_values):
    gini_scores = []
    for var in X.columns:
        if missing_values == "ignore":
            X_var_nonull = X[var].dropna()
            _auc = roc_auc_score(
                y[y.index.isin(X_var_nonull.index)],
                X_var_nonull,
                multi_class="ovr",
                average="weighted",
            )
        else:
            _auc = roc_auc_score(y, X[var], multi_class="ovr", average="weighted")

        score = 2 * _auc - 1
        gini_scores.append(score)

    gini_table = pd.DataFrame(gini_scores, index=X.columns, columns=["gini"])
    gini_table["gini"] = abs(gini_table["gini"].astype("float64"))

    return gini_table


def _calculate_corr_table(X, y):
    df = pd.concat([X, y], axis=1)
    corr_table = df.corr().unstack().reset_index(name="value")
    corr_table.rename(columns={"level_0": "col1", "level_1": "col2"}, inplace=True)
    corr_table = corr_table[corr_table["col1"] != corr_table["col2"]]
    corr_table["abs_value"] = abs(corr_table["value"])
    corr_table = corr_table[corr_table["col1"] != "TARGET"]
    target_corr = corr_table[corr_table["col2"] == "TARGET"]
    corr_table = corr_table[corr_table["col2"] != "TARGET"]
    target_corr.rename(columns={"abs_value": "gini"}, inplace=True)
    target_corr = target_corr.set_index("col1")
    target_corr = target_corr.drop(columns=["col2", "value"], axis=1, errors="ignore")
    target_corr["gini"] = target_corr["gini"].astype("float64")

    return corr_table, target_corr


def _prepare_compare_table(corr_table, gini_table, threshold=0.7):

    corr_table = corr_table[corr_table["abs_value"] >= threshold]
    corr_table["id"] = corr_table[["col1", "col2"]].min(axis=1) + corr_table[
        ["col1", "col2"]
    ].max(axis=1)

    compare_table = pd.merge(
        left=corr_table, right=gini_table, left_on="col1", right_index=True, how="left"
    ).rename(columns={"gini": "col1_gini"})

    compare_table = pd.merge(
        left=compare_table,
        right=gini_table,
        left_on="col2",
        right_index=True,
        how="left",
    ).rename(columns={"gini": "col2_gini"})

    compare_table = (
        compare_table.sort_values(["col1_gini", "col1", "col2"], ascending=False)
        .drop_duplicates("id")
        .sort_index()
    )
    compare_table.drop(columns=["id"], inplace=True)
    compare_table = compare_table.sort_values(
        ["col2_gini", "col2", "col1_gini"], ascending=False
    )

    return compare_table


def _eliminate_low_gini_correlated(compare_table):
    eleyen = compare_table["col1"].unique()
    elenen = compare_table["col2"].unique()

    kati_eleyen = list(set(eleyen) - set(elenen))
    kati_elenen = list(compare_table[(compare_table["col1"].isin(kati_eleyen))]["col2"])

    for i in range(0, len(compare_table)):
        compare_table2 = compare_table[
            (compare_table["col2"].isin(kati_eleyen) == False)
            & (compare_table["col2"].isin(kati_elenen) == False)
        ]

        if len(compare_table2) > 0:
            yeni_eleyen = compare_table2.iloc[0, 1]
            kati_eleyen.append(yeni_eleyen)
            kati_elenen = list(
                compare_table[(compare_table["col1"].isin(kati_eleyen))]["col2"]
            )
        else:
            kati_eleyen = list(set(kati_eleyen))
            kati_elenen = list(
                set(compare_table[(compare_table["col1"].isin(kati_eleyen))]["col2"])
            )
            break

    return set(kati_elenen)


class DropCorrelatedLowerGiniFeatures(BaseSelector):
    """
    DropCorrelatedLowGiniFeatures() finds and removes correlated features similar to DropCorrelatedFeatures(),
     however instead of first round first removed, it removes the feature with lower gini values.

    Parameters
    ----------
        variables: str or list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all numerical features in the dataset.

        threshold: float, default=0.8
        The correlation threshold above which a feature will be deemed correlated with another one and removed from the dataset.

        missing_values: str, default=ignore
        Whether the missing values should be raised as error or ignored when determining correlation. Takes values ‘raise’ and ‘ignore’.

        confirm_variables: bool, default=False
        If set to True, variables that are not present in the input dataframe will be removed from the list of variables.
         Only used when passing a variable list to the parameter variables. See parameter variables for more details.
    """
    def __init__(
        self,
        variables=None,
        threshold=0.9,
        missing_values="raise",
        confirm_variables=False,
    ):
        super().__init__(confirm_variables)

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values can only take ['raise','ignore']")

        self.threshold = threshold
        self.variables = _check_input_parameter_variables(variables)
        self.missing_values = missing_values

    def fit(self, X, y):
        # check input dataframe
        X = check_X(X)

        # find all numerical variables or check those entered are in the dataframe
        if self.variables is None:
            # Get all numerical variables by default
            self.variables_ = _find_numerical_variables(X)
        else:
            # Get user-specified variables
            if self.confirm_variables and _retain_variables_if_in_df is not None:
                variables = _retain_variables_if_in_df(X, self.variables)
            else:
                variables = self.variables

            # Filter to only numerical variables (intersect with numerical columns)
            numerical_vars = _find_numerical_variables(X)
            self.variables_ = [v for v in variables if v in numerical_vars]

        # save input features
        self._get_feature_names_in(X)

        # eliminate highly correlated features with lower gini
        self.gini_table = _calculate_gini_table(
            X[self.variables_], y, self.missing_values
        )
        self.corr_table, self.target_corr = _calculate_corr_table(X[self.variables_], y)
        self.compare_table = _prepare_compare_table(
            self.corr_table, self.gini_table, threshold=self.threshold
        )

        self.features_to_drop_ = _eliminate_low_gini_correlated(self.compare_table)

        return self

    def transform(self, X: pd.DataFrame, y=None):
        """
        Return dataframe with selected features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features].
            The input dataframe.

        y: Not used.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_selected_features]
            Pandas dataframe with the selected features.
        """

        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = check_X(X)

        # get selected features.
        self.selected_features = list(set(self.variables_) - set(self.features_to_drop_))

        for col in self.selected_features:
            if col not in X.columns:
                raise ValueError(f"Selected column: {col} not found in the input dataframe.")

        return X[self.selected_features]


    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        tags_dict["requires_y"] = True
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        return tags_dict

