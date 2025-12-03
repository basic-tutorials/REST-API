# -*- coding: utf-8 -*-

### WOE BINNIG FUNCTIONS
 
import pandas as pd
from optbinning import OptimalBinning
import copy


class opt_binning:

    def __init__(self, num_variables, cat_variables):
        self.num_binner = None
        self.cat_binner = None
        self.num_variables = num_variables
        self.cat_variables = cat_variables
        self.bin_pool = {}
        self.column_groups = []
        self.param_grid = None

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

    def fit(self, train, y, param_grid):

        self.param_grid = param_grid

        # change defaults
        self.param_grid["min_prebin_size"] = 0.01 if "min_prebin_size" not in self.param_grid else self.param_grid["min_prebin_size"]

        self._validate_param_grid(train)

        if "column_groups" in param_grid:
            self.column_groups = self.param_grid.pop("column_groups")

        if len(self.num_variables) > 0:
            for n in self.num_variables:
                column_params = self.param_grid.copy()
                column_params["dtype"] = "numerical"
                fit_params, _ = self._find_column_params(n)
                column_params.update(fit_params)  # overwrites colliding parameters

                binner = OptimalBinning(**column_params)
                self.bin_pool[n] = copy.deepcopy(binner.fit(train.loc[:, n], y))

        if len(self.cat_variables) > 0:
            for c in self.cat_variables:
                column_params = param_grid.copy()
                column_params["dtype"] = "categorical"
                fit_params, _ = self._find_column_params(c)
                column_params.update(fit_params)  # overwrites colliding parameters

                binner = OptimalBinning(**column_params)
                self.bin_pool[c] = copy.deepcopy(binner.fit(train.loc[:, c], y))

    def transform(self, train, y=None):

        if len(self.num_variables) > 0:
            for n in self.num_variables:
                binner = self.bin_pool[n]
                _, transform_params = self._find_column_params(n)
                train.loc[:, n] = binner.transform(train[n], metric='woe', metric_missing=transform_params["metric_missing"], show_digits=4)

        if len(self.cat_variables) > 0:
            for c in self.cat_variables:
                binner = self.bin_pool[c]
                _, transform_params = self._find_column_params(c)
                train.loc[:, c] = binner.transform(train[c], metric='woe', metric_missing=transform_params["metric_missing"], show_digits=4)

        return train

    def get_binning_table(self):

        keys = []
        binning_tables = []
        ginis = []
        for k, binner in self.bin_pool.items():
            keys.append(k)
            binning_table = binner.binning_table.build()
            binning_tables.append(binning_table.astype(str))
            ginis.append(binner.binning_table.gini)

        gini_table = pd.DataFrame(zip(keys, ginis), columns=["VARIABLE", "GINI_SCORE"])

        return keys, binning_tables, gini_table
