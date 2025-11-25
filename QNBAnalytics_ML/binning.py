# -*- coding: utf-8 -*-

### WOE BINNING using category_encoders.WOEEncoder

import pandas as pd
import numpy as np
import copy
import logging
from category_encoders.woe import WOEEncoder
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class opt_binning:
    """
    WOE binning using category_encoders.WOEEncoder
    Drop-in replacement for optbinning - compatible API
    """

    def __init__(self, num_variables, cat_variables):
        self.num_binner = None
        self.cat_binner = None
        self.num_variables = num_variables
        self.cat_variables = cat_variables
        self.bin_pool = {}
        self.column_groups = []
        self.param_grid = None
        self.woe_encoder = None  # Single encoder for all features
        self.bin_edges = {}  # Store bin edges for numeric features
        self.gini_scores = {}  # Store real Gini scores for each feature
        self.y_train = None  # Store target for Gini calculation

    def _validate_param_grid(self, train):
        if "column_groups" in self.param_grid:
            groups = self.param_grid["column_groups"]
            if not isinstance(groups, list):
                raise Exception("Column groups must be a list.")

            for g in groups:
                for c in g["columns"]:
                    if c not in list(train.columns):
                        raise Exception(f"{c} can not be found in the dataset.")

    def _find_column_params(self, column_name):
        fit_params = {}
        transform_params = {"metric_missing": "empirical"}

        for g in self.column_groups:
            if column_name in g["columns"]:
                group = g["params"].copy()

                # get and remove transform parameters from group
                transform_params = {
                    "metric_missing": group.pop("metric_missing") if "metric_missing" in group else "empirical",
                }

                # set fit params
                fit_params.update(group)

        return fit_params, transform_params

    def _bin_numeric_feature(self, x, y, var_name, bins=10, fit=True):
        """
        Create bins for numeric feature with consistent bin edges

        Parameters:
        -----------
        x : pd.Series
            Numeric feature
        y : pd.Series
            Target variable (not used currently)
        var_name : str
            Variable name for storing/retrieving bin edges
        bins : int
            Number of bins (only used during fit)
        fit : bool
            If True, create and store bin edges. If False, use stored edges.

        Returns:
        --------
        pd.Series: Binned feature (categorical)
        """
        if fit:
            # Create bins and store edges
            try:
                x_binned, bin_edges = pd.qcut(x, q=bins, duplicates='drop', labels=False, retbins=True)
                self.bin_edges[var_name] = bin_edges
            except:
                try:
                    x_binned, bin_edges = pd.cut(x, bins=bins, duplicates='drop', labels=False, retbins=True)
                    self.bin_edges[var_name] = bin_edges
                except:
                    # If all else fails, store None and return original
                    self.bin_edges[var_name] = None
                    x_binned = x
        else:
            # Use stored bin edges
            if var_name not in self.bin_edges or self.bin_edges[var_name] is None:
                return x

            try:
                x_binned = pd.cut(x, bins=self.bin_edges[var_name], labels=False, include_lowest=True)
            except:
                x_binned = x

        return x_binned

    def fit(self, train, y, param_grid):
        """
        Fit WOE encoding using category_encoders.WOEEncoder

        Parameters:
        -----------
        train : pd.DataFrame
            Training features
        y : pd.Series
            Target variable (binary: 0 or 1)
        param_grid : dict
            Parameters for binning
        """

        self.param_grid = param_grid

        # change defaults
        self.param_grid["min_prebin_size"] = 0.01 if "min_prebin_size" not in self.param_grid else self.param_grid["min_prebin_size"]

        self._validate_param_grid(train)

        if "column_groups" in param_grid:
            self.column_groups = self.param_grid.pop("column_groups")

        # Get all variables
        all_variables = list(self.num_variables) + list(self.cat_variables)

        if len(all_variables) == 0:
            return

        print(f"Fitting WOE encoding for {len(all_variables)} variables using category_encoders.WOEEncoder...")

        # Prepare data for WOE encoding
        train_binned = train.copy()

        # For numeric variables, create bins first
        if len(self.num_variables) > 0:
            for var in self.num_variables:
                if var in train_binned.columns:
                    # Bin numeric features into categories
                    max_bins = self.param_grid.get('max_n_prebins', 20)
                    train_binned[var] = self._bin_numeric_feature(train_binned[var], y, var_name=var, bins=max_bins, fit=True)

        # Now use WOEEncoder on all variables (numeric are now binned, categorical as-is)
        # Initialize WOEEncoder
        self.woe_encoder = WOEEncoder(
            cols=all_variables,
            randomized=True,  # Add random noise to prevent overfitting
            sigma=0.05,       # Small noise for stability
            regularization=1.0,  # Regularization parameter
            random_state=42
        )

        # Fit the encoder
        self.woe_encoder.fit(train_binned[all_variables], y)

        # Store target for Gini calculation
        self.y_train = y

        # Transform training data to calculate real Gini scores
        train_woe = self.woe_encoder.transform(train_binned[all_variables])

        # Calculate real Gini (2 * AUC - 1) for each feature
        print("Calculating Gini scores for all features...")
        for var in all_variables:
            try:
                # Get WOE-transformed values
                woe_values = train_woe[var].dropna()
                y_clean = y[woe_values.index]

                # Calculate AUC
                auc = roc_auc_score(y_clean, woe_values)
                # Calculate Gini from AUC
                gini = abs(2 * auc - 1)
                self.gini_scores[var] = gini
            except Exception as e:
                logger.warning(f"Could not calculate Gini for {var}: {str(e)}")
                self.gini_scores[var] = 0.0

        # Store binning information for each variable
        for var in all_variables:
            self.bin_pool[var] = {
                'encoder': self.woe_encoder,
                'is_numeric': var in self.num_variables,
                'binned_data': train_binned[var] if var in self.num_variables else None
            }

        print(f"WOE encoding fitted for {len(self.bin_pool)} variables")

    def transform(self, train, y=None):
        """
        Transform features using fitted WOE encoder

        Parameters:
        -----------
        train : pd.DataFrame
            Features to transform
        y : pd.Series (optional)
            Target variable (not used in transform)

        Returns:
        --------
        pd.DataFrame
            Transformed features with WOE values
        """

        if not self.woe_encoder:
            raise Exception("WOE encoder not fitted. Call fit() first.")

        train_transformed = train.copy()

        # For numeric variables, apply same binning as during fit using stored edges
        for var in self.num_variables:
            if var in train_transformed.columns and var in self.bin_pool:
                # Apply stored bin edges (fit=False means use stored edges)
                train_transformed[var] = self._bin_numeric_feature(train_transformed[var], None, var_name=var, bins=None, fit=False)

        # Apply WOE encoding
        all_variables = [col for col in train_transformed.columns if col in self.bin_pool]

        if len(all_variables) > 0:
            train_transformed[all_variables] = self.woe_encoder.transform(train_transformed[all_variables])

        return train_transformed

    def get_binning_table(self):
        """
        Get binning tables and Gini scores

        Returns:
        --------
        tuple: (keys, binning_tables, gini_table)
            - keys: list of variable names
            - binning_tables: list of DataFrames with binning details
            - gini_table: DataFrame with variable names and Gini scores
        """

        if not self.bin_pool:
            raise Exception("WOE encoder not fitted. Call fit() first.")

        keys = []
        binning_tables = []
        ginis = []

        for var_name, bin_info in self.bin_pool.items():
            keys.append(var_name)

            # Get WOE mapping from encoder
            # category_encoders stores mapping as a dictionary where keys are column names
            var_mapping = None

            if hasattr(self.woe_encoder, 'mapping') and self.woe_encoder.mapping:
                try:
                    # mapping is a dict with column names as keys and Series as values
                    if isinstance(self.woe_encoder.mapping, dict) and var_name in self.woe_encoder.mapping:
                        var_mapping = self.woe_encoder.mapping[var_name]
                    else:
                        logger.warning(f"Variable {var_name} not found in WOE encoder mapping")
                except Exception as e:
                    logger.error(f"Error accessing WOE mapping for {var_name}: {str(e)}")

            # Create binning table
            if var_mapping is not None and hasattr(var_mapping, 'index') and hasattr(var_mapping, 'values'):
                # var_mapping is a pandas Series
                binning_df = pd.DataFrame({
                    'bin': var_mapping.index.astype(str),
                    'woe': var_mapping.values
                })
            else:
                # Fallback: create simple binning table
                logger.warning(f"Using fallback binning table for {var_name} - WOE mapping not found")
                binning_df = pd.DataFrame({'bin': [var_name], 'woe': [0]})

            binning_tables.append(binning_df.astype(str))

            # Use real Gini score calculated during fit
            if var_name in self.gini_scores:
                gini = self.gini_scores[var_name]
            else:
                logger.warning(f"No Gini score found for {var_name}, using 0.0")
                gini = 0.0

            ginis.append(gini)

        gini_table = pd.DataFrame(zip(keys, ginis), columns=["VARIABLE", "GINI_SCORE"])

        return keys, binning_tables, gini_table
