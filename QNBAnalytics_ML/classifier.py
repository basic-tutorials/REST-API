import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LogisticRegression2
from lightgbm import LGBMClassifier, log_evaluation
from xgboost import XGBClassifier

from QNBAnalytics_ML.helpers import getLogger
from QNBAnalytics_ML.feature_importances import get_feature_importances, plot_feature_importances
import QNBAnalytics_ML.default_grids as default_grids


from datetime import datetime
import random
import os
random.seed(48)
np.random.seed(48)


if os.getenv("MLFLOW_LOGGING"):
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_URL", 'http://127.0.0.1:5000'))



class model_args():

    @staticmethod
    def main_targeter(y_train):
        logger = getLogger()
        target_vals = list(pd.unique(y_train))
        target_vals.sort()

        if y_train.dtypes == "object":
            target_type = "Categoric"
        else:
            target_type = "Numeric"

        if target_vals == [0, 1]:
            target_def = "Binary"
        else:
            target_def = "Non-binary"

        if (target_type == "Numeric") & (target_def == "Binary"):
            logger.info("This is a binary classification problem.\n")

            if y_train.value_counts().min() / len(y_train) <= 0.01:
                fit_params = {'class_weight': 'balanced'}
                scoring = {'scoring': 'f1'}
            else:
                fit_params = {'class_weight': None}
                scoring = {'scoring': 'roc_auc'}

        return fit_params, scoring

    @staticmethod
    def main_sampler(x_train, train_size=None, random_state=48):
        all_index = x_train.index.to_list()
        all_val_index = x_train.index.to_series().sample(len(all_index)//10, random_state=random_state).to_list()
        all_train_index = list(set(all_index) - set(all_val_index))

        smpl_index = []
        smpl_train_index = []
        smpl_val_index = []

        if train_size is not None:
            smpl_index = x_train.sample(n=train_size, random_state=random_state).index.to_list()
            smpl_val_index = pd.Series(smpl_index).sample(len(smpl_index)//10, random_state=random_state)
            smpl_train_index = list(set(smpl_index) - set(smpl_val_index))

        return sorted(all_index), sorted(all_train_index), sorted(all_val_index), sorted(smpl_index), \
            sorted(smpl_train_index), sorted(smpl_val_index)


class LogisticRegression:
    model_name = "logistic_regression"

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

        self.model = None
        self.train_gini = None
        self.cv_gini = None
        self.valid_gini = None
        self.test_gini = None
        self.train_prob = None
        self.train_size = train_size
        self.param_opt = param_opt
        self.param_opt_method = param_opt_method
        self.baseEstimator = baseEstimator
        self.random_state = random_state
        self.nprocs = nprocs
        self.n_iter = n_iter
        self.grid = param_grid
        self.cv = cv
        self.verbose = verbose
        self.refit_to_all = refit_to_all
        self.alpha = alpha
        self.model_perf = pd.DataFrame(columns=['Model Name', 'Phase', 'Train Gini', 'Valid Gini', 'Test Gini'])
        self.cv_results = None
        self.time_limit = time_limit
        self.n_feature_params = n_feature_params
        self.importance_direction_type = importance_direction_type
        self.targetcorr_df = targetcorr_df

        logger = getLogger()
        
        if (self.grid is None) and (self.param_opt_method == "RandomizedSearchCV"):
            self.grid = default_grids.grids[self.model_name]
        
    def select_function(self, cv, train):
        cv = np.array(cv)
        train = np.array(train)

        return cv - (train - cv) * self.alpha

    def fit_model(self, x_train, y_train):
        logger = getLogger()
        logger.info("\n")
        logger.info("Logistic Regression fitting started.\n")

        self.start = datetime.now()

        fit_params, scoring = model_args.main_targeter(y_train)

        self.model = LogisticRegression2(solver='lbfgs', class_weight=fit_params.get('class_weight'),
                                         random_state=self.random_state)

        self.all_index, self.all_train_index, self.all_val_index, self.smpl_index, self.smpl_train_index, self.smpl_val_index = model_args.main_sampler(
            x_train=x_train, train_size=self.train_size, random_state=self.random_state)

        if self.param_opt == False:

            if self.train_size != None:
                self.model.fit(x_train.loc[self.smpl_index], y_train.loc[self.smpl_index])
            else:
                self.model.fit(x_train.loc[self.all_index], y_train.loc[self.all_index])


        elif self.param_opt == True:
            logger.info("Parameter optimization started.\n")

            refit = True


            if self.param_opt_method == "RandomizedSearchCV":
                self.model = RandomizedSearchCV(
                    estimator=self.model,
                    param_distributions=self.grid,
                    cv=self.cv,
                    n_iter=self.n_iter,
                    refit=refit,
                    scoring=scoring.get('scoring'),
                    n_jobs=self.nprocs,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    return_train_score=True
                )
            else:
                logger.info(
                    "RandomizedSearchCV optimization methods should be selected.\n")

            if self.train_size != None:
                self.model.fit(x_train.loc[self.smpl_index], y_train.loc[self.smpl_index])
                self.best_estimator = self.model.best_estimator_
                self.cv_gini = 2 * (self.model.best_score_) - 1
                self.model = self.best_estimator

                if self.refit_to_all == True:
                    logger.info("Refitting to all data...")
                    self.model.fit(x_train.loc[self.all_index], y_train.loc[self.all_index])

                elif self.alpha != None:
                    self.model.fit(x_train.loc[self.smpl_index], y_train.loc[self.smpl_index])

            else:
                self.model.fit(x_train.loc[self.all_index], y_train.loc[self.all_index])

                self.best_estimator = self.model.best_estimator_
                self.cv_gini = 2 * (self.model.best_score_) - 1
                self.model = self.best_estimator

            logger.info("Results of %s" % self.best_estimator)

            logger.info("Mean CV score:  %s" % self.cv_gini)

        try:
            logger.info("Calculating SHAP feature importances...")
            feature_imp_df = get_feature_importances(self.model, x_train, x_train.sample(
                n=1000 if len(x_train.index) > 1000 else len(x_train.index), random_state=0), 
                correlation_type=self.importance_direction_type, corr_df=self.targetcorr_df)

            # Putting it before plot incase exception happens there
            if os.getenv("MLFLOW_LOGGING"):
                feature_imp_df.to_csv("feature_imp.csv")
                mlflow.log_artifact("feature_imp.csv")
                os.remove("feature_imp.csv")

            fig = plot_feature_importances(feature_imp_df, n=self.n_feature_params)
            self.feature_importances = feature_imp_df
            if os.getenv("MLFLOW_LOGGING"):
                with open("feature_imp.png", "wb") as f:
                    fig.savefig(f)
                    mlflow.log_artifact("feature_imp.png")
                    os.remove("feature_imp.png")
        except Exception as e:
            logger.error("Feature importance calculation failed. %s" % e, exc_info=True)
            self.feature_importances = None

        cre_time = datetime.now()
        self.creation_ = str(cre_time)
        id_1 = str(round((self.start - datetime(1970, 1, 1)).total_seconds()))
        id_2 = str(round((cre_time - datetime(1970, 1, 1)).total_seconds()))
        rndm = str(random.randint(1000, 9999))
        self.id_ = id_1 + '-' + rndm + '-' + id_2
        self.train_time_ = (datetime.now() - self.start).total_seconds()

        logger.info("Model fitting completed.\n")

        self.train_prob = self.model.predict_proba(x_train)[:, 1]
        self.train_gini = 2 * roc_auc_score(y_train, self.train_prob) - 1
        logger.info("With ROC: " % roc_auc_score(y_train, self.train_prob))
        logger.info("With gini: " % (self.train_gini))

    def predict(self, x_test, y_test=[], sample='Validation'):
        logger = getLogger()
        x_test = x_test[self.model.feature_names_in_]
        self.test_pred_prob = self.model.predict_proba(x_test)[:, 1]
        self.test_pred_prob = np.round(self.test_pred_prob,38)

        test_pred = self.model.predict(x_test)

        sample_gini = None
        if len(y_test) >= 1:
            if os.getenv("MLFLOW_LOGGING"):
                eval_data = x_test.copy()
                eval_data['label'] = y_test.tolist()
                try:
                    model_uri = mlflow.get_artifact_uri('best_estimator')
                    mlflow.evaluate(
                        model_uri,
                        eval_data,
                        targets='label',
                        feature_names=x_test.columns.tolist(),
                        model_type="regressor",
                        dataset_name="testset",

                    )
                except Exception as e:
                    logger.info(e)
            if sample == 'Validation':
                self.valid_gini = 2 * roc_auc_score(y_test, self.test_pred_prob) - 1
            elif sample == 'Test':
                self.test_gini = 2 * roc_auc_score(y_test, self.test_pred_prob) - 1
            else:
                sample_gini = 2 * roc_auc_score(y_test, self.test_pred_prob) - 1

            self.model_perf = {'Model Name': 'Logistic Regression', 'Phase': 'Base', 'Train Gini': self.train_gini,
                               'CV Gini': self.cv_gini, 'Valid Gini': self.valid_gini, 'Test Gini': self.test_gini}

            logger.info("Results of Logistic Regression")

            logger.info("Train gini: %s" % self.train_gini)
            logger.info("CV gini: %s" % self.cv_gini)
            logger.info("Valid gini: %s" % self.valid_gini)
            logger.info("Test gini: %s\n" % self.test_gini)
            if os.getenv("MLFLOW_LOGGING"):
                if self.train_gini:
                    mlflow.log_metric("train_gini", self.train_gini)
                if self.cv_gini:
                    mlflow.log_metric("cv_gini", self.cv_gini)
                if self.valid_gini:
                    mlflow.log_metric("valid_gini", self.valid_gini)
                if self.test_gini:
                    mlflow.log_metric("test_gini", self.test_gini)

        return self.test_pred_prob, test_pred

    def model_performance(self):
        logger = getLogger()
        logger.info("Results of Logistic Regression")

        logger.info("Train gini: %s" % self.train_gini)
        logger.info("CV gini: %s" % self.cv_gini)
        logger.info("Valid gini: %s" % self.valid_gini)
        logger.info("Test gini: %s\n" % self.test_gini)

        return self.model_perf


class RandomForest:
    model_name = "random_forest_classifier"
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

        self.model = None
        self.train_gini = None
        self.cv_gini = None
        self.valid_gini = None
        self.test_gini = None
        self.train_prob = None
        self.train_size = train_size
        self.param_opt = param_opt
        self.param_opt_method = param_opt_method
        self.baseEstimator = baseEstimator
        self.random_state = random_state
        self.nprocs = nprocs
        self.n_iter = n_iter
        self.grid = param_grid
        self.cv = cv
        self.verbose = verbose
        self.refit_to_all = refit_to_all
        self.alpha = alpha
        self.model_perf = pd.DataFrame(columns=['Model Name', 'Phase', 'Train Gini', 'Valid Gini', 'Test Gini'])
        self.cv_results = None
        self.time_limit = time_limit
        self.n_feature_params = n_feature_params
        self.importance_direction_type = importance_direction_type
        self.targetcorr_df = targetcorr_df
        logger = getLogger()

        if (self.grid is None) and (self.param_opt_method == "RandomizedSearchCV"):
            self.grid = default_grids.grids[self.model_name]

    def select_function(self, cv, train):
        cv = np.array(cv)
        train = np.array(train)

        return cv - (train - cv) * self.alpha

    def fit_model(self, x_train, y_train):
        logger = getLogger()
        logger.info("\n")
        logger.info("Random Forest fitting started.\n")

        self.start = datetime.now()

        fit_params, scoring = model_args.main_targeter(y_train)

        self.model = RandomForestClassifier(max_depth=10, n_estimators=200, class_weight=fit_params.get('class_weight'),
                                            random_state=self.random_state)
        self.all_index, self.all_train_index, self.all_val_index, self.smpl_index, self.smpl_train_index, self.smpl_val_index = model_args.main_sampler(
            x_train=x_train, train_size=self.train_size, random_state=self.random_state)

        if self.param_opt == False:

            if self.train_size != None:
                self.model.fit(x_train.loc[self.smpl_index], y_train.loc[self.smpl_index])
            else:
                self.model.fit(x_train.loc[self.all_index], y_train.loc[self.all_index])


        elif self.param_opt == True:
            logger.info("Parameter optimization started.\n")

            refit = True

            if self.param_opt_method == "RandomizedSearchCV":
                self.model = RandomizedSearchCV(
                    estimator=self.model,
                    param_distributions=self.grid,
                    cv=self.cv,
                    n_iter=self.n_iter,
                    refit=refit,
                    scoring=scoring.get('scoring'),
                    n_jobs=self.nprocs,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    return_train_score=True
                )
            else:
                logger.info(
                    "RandomizedSearchCV optimization methods should be selected.\n")

            if self.train_size != None:
                self.model.fit(x_train.loc[self.smpl_index], y_train.loc[self.smpl_index])

                self.best_estimator = self.model.best_estimator_
                self.cv_gini = 2 * (self.model.best_score_) - 1
                self.model = self.best_estimator

                if self.refit_to_all == True:
                    logger.info("Refitting to all data...")
                    self.model.fit(x_train.loc[self.all_index], y_train.loc[self.all_index])

                elif self.alpha != None:
                    self.model.fit(x_train.loc[self.smpl_index], y_train.loc[self.smpl_index])

            else:
                self.model.fit(x_train.loc[self.all_index], y_train.loc[self.all_index])

                self.best_estimator = self.model.best_estimator_
                self.cv_gini = 2 * (self.model.best_score_) - 1
                self.model = self.best_estimator

            logger.info("Results of %s" % self.best_estimator)

            logger.info("Mean CV score:  %s" % self.cv_gini)

        try:
            logger.info("Calculating SHAP feature importances...")
            feature_imp_df = get_feature_importances(self.model, x_train, x_train.sample(
                n=1000 if len(x_train.index) > 1000 else len(x_train.index), random_state=0),
                correlation_type=self.importance_direction_type, corr_df=self.targetcorr_df)

            # Putting it before plot incase exception happens there
            if os.getenv("MLFLOW_LOGGING"):
                feature_imp_df.to_csv("feature_imp.csv")
                mlflow.log_artifact("feature_imp.csv")
                os.remove("feature_imp.csv")

            fig = plot_feature_importances(feature_imp_df, n=self.n_feature_params)
            self.feature_importances = feature_imp_df
            if os.getenv("MLFLOW_LOGGING"):
                with open("feature_imp.png", "wb") as f:
                    fig.savefig(f)
                    mlflow.log_artifact("feature_imp.png")
                    os.remove("feature_imp.png")
        except Exception as e:
            logger.info("Feature importance calculation failed. %s" % e)
            self.feature_importances = None
        cre_time = datetime.now()
        self.creation_ = str(cre_time)
        id_1 = str(round((self.start - datetime(1970, 1, 1)).total_seconds()))
        id_2 = str(round((cre_time - datetime(1970, 1, 1)).total_seconds()))
        rndm = str(random.randint(1000, 9999))
        self.id_ = id_1 + '-' + rndm + '-' + id_2
        self.train_time_ = (datetime.now() - self.start).total_seconds()

        logger.info("Model fitting completed.\n")

        self.train_prob = self.model.predict_proba(x_train)[:, 1]
        self.train_gini = 2 * roc_auc_score(y_train, self.train_prob) - 1
        logger.info("With ROC: " % roc_auc_score(y_train, self.train_prob))
        logger.info("With gini: " % (self.train_gini))

    def predict(self, x_test, y_test=[], sample='Validation'):
        logger = getLogger()
        x_test = x_test[self.model.feature_names_in_]
        self.test_pred_prob = self.model.predict_proba(x_test)[:, 1]
        self.test_pred_prob = np.round(self.test_pred_prob,38)

        test_pred = self.model.predict(x_test)

        sample_gini = None
        if len(y_test) >= 1:
            if os.getenv("MLFLOW_LOGGING"):
                eval_data = x_test.copy()
                eval_data['label'] = y_test.tolist()
                try:
                    model_uri = mlflow.get_artifact_uri('best_estimator')
                    mlflow.evaluate(
                        model_uri,
                        eval_data,
                        targets='label',
                        feature_names=x_test.columns.tolist(),
                        model_type="classification",
                        dataset_name="testset",

                    )
                except Exception as e:
                    logger.info(e)
            if sample == 'Validation':
                self.valid_gini = 2 * roc_auc_score(y_test, self.test_pred_prob) - 1
            elif sample == 'Test':
                self.test_gini = 2 * roc_auc_score(y_test, self.test_pred_prob) - 1
            else:
                sample_gini = 2 * roc_auc_score(y_test, self.test_pred_prob) - 1

            self.model_perf = {'Model Name': 'Random Forest', 'Phase': 'Base', 'Train Gini': self.train_gini,
                               'CV Gini': self.cv_gini, 'Valid Gini': self.valid_gini, 'Test Gini': self.test_gini}

            logger.info("Results of Random Forest")

            logger.info("Train gini: %s" % self.train_gini)
            logger.info("CV gini: %s" % self.cv_gini)
            logger.info("Valid gini: %s" % self.valid_gini)
            logger.info("Test gini: %s\n" % self.test_gini)
            if os.getenv("MLFLOW_LOGGING"):
                if self.train_gini:
                    mlflow.log_metric("train_gini", self.train_gini)
                if self.cv_gini:
                    mlflow.log_metric("cv_gini", self.cv_gini)
                if self.valid_gini:
                    mlflow.log_metric("valid_gini", self.valid_gini)
                if self.test_gini:
                    mlflow.log_metric("test_gini", self.test_gini)

        return self.test_pred_prob, test_pred

    def model_performance(self):
        logger = getLogger()
        logger.info("Results of Random Forest")

        logger.info("Train gini: %s" % self.train_gini)
        logger.info("CV gini: %s" % self.cv_gini)
        logger.info("Valid gini: %s" % self.valid_gini)
        logger.info("Test gini: %s\n" % self.test_gini)

        return self.model_perf


class LGBM:
    model_name = "lightgbm_classifier"
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

        self.model = None
        self.train_gini = None
        self.cv_gini = None
        self.valid_gini = None
        self.test_gini = None
        self.train_prob = None
        self.train_size = train_size
        self.param_opt = param_opt
        self.param_opt_method = param_opt_method
        self.baseEstimator = baseEstimator
        self.random_state = random_state
        self.nprocs = nprocs
        self.n_iter = n_iter
        self.grid = param_grid
        self.cv = cv
        self.verbose = verbose
        self.refit_to_all = refit_to_all
        self.alpha = alpha
        self.model_perf = pd.DataFrame(columns=['Model Name', 'Phase', 'Train Gini', 'Valid Gini', 'Test Gini'])
        self.cv_results = None
        self.time_limit = time_limit
        self.n_feature_params = n_feature_params
        self.importance_direction_type = importance_direction_type
        self.targetcorr_df = targetcorr_df

        logger = getLogger()

        if (self.grid is None) and (self.param_opt_method == "RandomizedSearchCV"):
            self.grid = default_grids.grids[self.model_name]
        
    def select_function(self, cv, train):
        cv = np.array(cv)
        train = np.array(train)

        return cv - (train - cv) * self.alpha

    def fit_model(self, x_train, y_train):
        logger = getLogger()
        logger.info("\n")
        logger.info("LGBM fitting started.\n")

        self.start = datetime.now()

        fit_params, scoring = model_args.main_targeter(y_train)

        self.model = LGBMClassifier(n_jobs=self.nprocs, class_weight=fit_params.get('class_weight'), random_state=self.random_state)
        self.all_index, self.all_train_index, self.all_val_index, self.smpl_index, self.smpl_train_index, self.smpl_val_index = model_args.main_sampler(
            x_train=x_train, train_size=self.train_size, random_state=self.random_state)

        early_stop = 0
        ea_list = ['early_stopping_round', 'early_stopping_rounds', 'early_stopping', 'n_iter_no_change']
        if any(name in ea_list for name in self.grid):
            early_stop = 1

        if self.param_opt == False:

            if self.train_size != None:
                if early_stop == 1:
                    eval_set = (x_train.loc[self.smpl_val_index], y_train.loc[self.smpl_val_index])
                    self.model.fit(x_train.loc[self.smpl_train_index], y_train.loc[self.smpl_train_index],
                                   eval_metric='auc', eval_set=eval_set, callbacks=[log_evaluation(self.verbose)])
                else:
                    self.model.fit(x_train.loc[self.smpl_index], y_train.loc[self.smpl_index], callbacks=[log_evaluation(self.verbose)])
            else:
                if early_stop == 1:
                    eval_set = (x_train.loc[self.all_val_index], y_train.loc[self.all_val_index])
                    self.model.fit(x_train.loc[self.all_train_index], y_train.loc[self.all_train_index],
                                   eval_metric='auc', eval_set=eval_set, callbacks=[log_evaluation(self.verbose)])
                else:
                    self.model.fit(x_train.loc[self.all_index], y_train.loc[self.all_index], callbacks=[log_evaluation(self.verbose)])


        elif self.param_opt == True:
            logger.info("Parameter optimization started.\n")

            #if self.train_size is not None and self.refit_to_all == True:
            #    refit=False
            #else:
            #    refit=True
            refit = True

            if self.param_opt_method == "RandomizedSearchCV":
                self.model = RandomizedSearchCV(
                    estimator=self.model,
                    param_distributions=self.grid,
                    cv=self.cv,
                    n_iter=self.n_iter,
                    refit=refit,
                    scoring=scoring.get('scoring'),
                    n_jobs=self.nprocs,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    return_train_score=True
                )
            else:
                logger.info(
                    "RandomizedSearchCV optimization methods should be selected.\n")

            if self.train_size != None:
                if early_stop == 1:
                    eval_set = (x_train.loc[self.smpl_val_index], y_train.loc[self.smpl_val_index])
                    self.model.fit(x_train.loc[self.smpl_train_index], y_train.loc[self.smpl_train_index],
                                   eval_metric='auc', eval_set=eval_set, callbacks=[log_evaluation(self.verbose)])

                    self.best_estimator = self.model.best_estimator_
                    self.cv_gini = 2 * (self.model.best_score_) - 1
                    self.model = self.best_estimator

                    if self.refit_to_all == True:

                        logger.info("Refitting to all data...")
                        eval_set = (x_train.loc[self.all_val_index], y_train.loc[self.all_val_index])
                        self.model.fit(x_train.loc[self.all_train_index], y_train.loc[self.all_train_index],
                                       eval_metric='auc', eval_set=eval_set, callbacks=[log_evaluation(self.verbose)])

                    elif self.alpha != None:
                        eval_set = (x_train.loc[self.smpl_val_index], y_train.loc[self.smpl_val_index])
                        self.model.fit(x_train.loc[self.smpl_train_index], y_train.loc[self.smpl_train_index],
                                       eval_metric='auc', eval_set=eval_set, callbacks=[log_evaluation(self.verbose)])

                elif early_stop == 0:
                    self.model.fit(x_train.loc[self.smpl_index], y_train.loc[self.smpl_index], callbacks=[log_evaluation(self.verbose)])

                    self.best_estimator = self.model.best_estimator_
                    self.cv_gini = 2 * (self.model.best_score_) - 1
                    self.model = self.best_estimator

                    if self.refit_to_all == True:

                        logger.info("Refitting to all data...")
                        self.model.fit(x_train.loc[self.all_index], y_train.loc[self.all_index], callbacks=[log_evaluation(self.verbose)])

                    elif self.alpha != None:

                        self.model.fit(x_train.loc[self.smpl_index], y_train.loc[self.smpl_index], callbacks=[log_evaluation(self.verbose)])

            else:

                if early_stop == 1:
                    eval_set = (x_train.loc[self.all_val_index], y_train.loc[self.all_val_index])
                    self.model.fit(x_train.loc[self.all_train_index], y_train.loc[self.all_train_index],
                                   eval_metric='auc', eval_set=eval_set, callbacks=[log_evaluation(self.verbose)])

                    self.best_estimator = self.model.best_estimator_
                    self.cv_gini = 2 * (self.model.best_score_) - 1
                    self.model = self.best_estimator


                elif early_stop == 0:
                    self.model.fit(x_train.loc[self.all_index], y_train.loc[self.all_index], callbacks=[log_evaluation(self.verbose)])

                    self.best_estimator = self.model.best_estimator_
                    self.cv_gini = 2 * (self.model.best_score_) - 1
                    self.model = self.best_estimator

            logger.info("Results of %s" % self.best_estimator)

            logger.info("Mean CV score:  %s" % self.cv_gini)

        try:
            logger.info("Calculating SHAP feature importances...")
            feature_imp_df = get_feature_importances(self.model, x_train, x_train.sample(
                n=1000 if len(x_train.index) > 1000 else len(x_train.index), random_state=0),
                correlation_type=self.importance_direction_type, corr_df=self.targetcorr_df)

            # Putting it before plot incase exception happens there
            if os.getenv("MLFLOW_LOGGING"):
                feature_imp_df.to_csv("feature_imp.csv")
                mlflow.log_artifact("feature_imp.csv")
                os.remove("feature_imp.csv")

            fig = plot_feature_importances(feature_imp_df, n=self.n_feature_params)
            self.feature_importances = feature_imp_df
            if os.getenv("MLFLOW_LOGGING"):
                with open("feature_imp.png", "wb") as f:
                    fig.savefig(f)
                    mlflow.log_artifact("feature_imp.png")
                    os.remove("feature_imp.png")
        except Exception as e:
            logger.info("Feature importance calculation failed. %s" % e)
            self.feature_importances = None

        if early_stop:
            del eval_set

        cre_time = datetime.now()
        self.creation_ = str(cre_time)
        id_1 = str(round((self.start - datetime(1970, 1, 1)).total_seconds()))
        id_2 = str(round((cre_time - datetime(1970, 1, 1)).total_seconds()))
        rndm = str(random.randint(1000, 9999))
        self.id_ = id_1 + '-' + rndm + '-' + id_2
        self.train_time_ = (datetime.now() - self.start).total_seconds()

        logger.info("Model fitting completed.\n")

        self.train_prob = self.model.predict_proba(x_train)[:, 1]
        self.train_gini = 2 * roc_auc_score(y_train, self.train_prob) - 1
        logger.info("With ROC: " % roc_auc_score(y_train, self.train_prob))
        logger.info("With gini: " % (self.train_gini))

    def predict(self, x_test, y_test=[], sample='Validation'):
        logger = getLogger()
        x_test = x_test[self.model.feature_name_]
        self.test_pred_prob = self.model.predict_proba(x_test)[:, 1]
        self.test_pred_prob = np.round(self.test_pred_prob,38)

        test_pred = self.model.predict(x_test)

        sample_gini = None
        if len(y_test) >= 1:
            if os.getenv("MLFLOW_LOGGING"):
                eval_data = x_test.copy()
                eval_data['label'] = y_test.tolist()
                try:
                    model_uri = mlflow.get_artifact_uri('best_estimator')
                    mlflow.evaluate(
                        model_uri,
                        eval_data,
                        targets='label',
                        feature_names=x_test.columns.tolist(),
                        model_type="classification",
                        dataset_name="testset",
                    )
                except Exception as e:
                    logger.info(e)
            if sample == 'Validation':
                self.valid_gini = 2 * roc_auc_score(y_test, self.test_pred_prob) - 1
            elif sample == 'Test':
                self.test_gini = 2 * roc_auc_score(y_test, self.test_pred_prob) - 1
            else:
                sample_gini = 2 * roc_auc_score(y_test, self.test_pred_prob) - 1

            self.model_perf = {'Model Name': 'LGBM', 'Phase': 'Base', 'Train Gini': self.train_gini,
                               'CV Gini': self.cv_gini, 'Valid Gini': self.valid_gini, 'Test Gini': self.test_gini}

            logger.info("Results of LGBM")

            logger.info("Train gini: %s" % self.train_gini)
            logger.info("CV gini: %s" % self.cv_gini)
            logger.info("Valid gini: %s" % self.valid_gini)
            logger.info("Test gini: %s\n" % self.test_gini)
            if os.getenv("MLFLOW_LOGGING"):
                if self.train_gini:
                    mlflow.log_metric("train_gini", self.train_gini)
                if self.cv_gini:
                    mlflow.log_metric("cv_gini", self.cv_gini)
                if self.valid_gini:
                    mlflow.log_metric("valid_gini", self.valid_gini)
                if self.test_gini:
                    mlflow.log_metric("test_gini", self.test_gini)

        return self.test_pred_prob, test_pred

    def model_performance(self):
        logger = getLogger()
        logger.info("Results of LGBM")

        logger.info("Train gini: %s" % self.train_gini)
        logger.info("CV gini: %s" % self.cv_gini)
        logger.info("Valid gini: %s" % self.valid_gini)
        logger.info("Test gini: %s\n" % self.test_gini)

        return self.model_perf


class XGBoost:
    model_name = "xgboost_classifier"
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

        self.model = None
        self.train_gini = None
        self.cv_gini = None
        self.valid_gini = None
        self.test_gini = None
        self.train_prob = None
        self.train_size = train_size
        self.param_opt = param_opt
        self.param_opt_method = param_opt_method
        self.baseEstimator = baseEstimator
        self.random_state = random_state
        self.nprocs = nprocs
        self.n_iter = n_iter
        self.grid = param_grid
        self.cv = cv
        self.verbose = verbose
        self.refit_to_all = refit_to_all
        self.alpha = alpha
        self.model_perf = pd.DataFrame(columns=['Model Name', 'Phase', 'Train Gini', 'Valid Gini', 'Test Gini'])
        self.cv_results = None
        self.time_limit = time_limit
        self.n_feature_params = n_feature_params
        self.importance_direction_type = importance_direction_type
        self.targetcorr_df = targetcorr_df
        logger = getLogger()


        if (self.grid is None) and (self.param_opt_method == "RandomizedSearchCV"):
            self.grid = default_grids.grids[self.model_name]
       

    def select_function(self, cv, train):
        cv = np.array(cv)
        train = np.array(train)

        return cv - (train - cv) * self.alpha

    def fit_model(self, x_train, y_train):
        logger = getLogger()
        logger.info("\n")
        logger.info("XGBoost fitting started.\n")

        self.start = datetime.now()

        fit_params, scoring = model_args.main_targeter(y_train)

        self.model = XGBClassifier(class_weight=fit_params.get('class_weight'), random_state=self.random_state)

        self.all_index, self.all_train_index, self.all_val_index, self.smpl_index, self.smpl_train_index, self.smpl_val_index = model_args.main_sampler(
            x_train=x_train, train_size=self.train_size, random_state=self.random_state)

        early_stop = 0
        ea_list = ['early_stopping_round', 'early_stopping_rounds', 'early_stopping', 'n_iter_no_change']
        if any(name in ea_list for name in self.grid):
            early_stop = 1
            rounds = self.grid.get('early_stopping_rounds', 10)

            
        my_dict = self.grid.copy()

        if self.param_opt == False:

            if self.train_size != None:
                if early_stop == 1:
                    # eval_set = [(x_train.loc[self.smpl_train_index],y_train.loc[self.smpl_train_index]),(x_train.loc[self.smpl_val_index],y_train.loc[self.smpl_val_index])]
                    eval_set = [(x_train.loc[self.smpl_val_index], y_train.loc[self.smpl_val_index])]
                    self.model.fit(x_train.loc[self.smpl_train_index], y_train.loc[self.smpl_train_index],
                                   eval_set=eval_set,
                                   verbose=self.verbose)
                else:
                    self.model.fit(x_train.loc[self.smpl_index], y_train.loc[self.smpl_index], verbose=self.verbose)
            else:
                if early_stop == 1:
                    # eval_set = [(x_train.loc[self.all_train_index],y_train.loc[self.all_train_index]),(x_train.loc[self.all_val_index],y_train.loc[self.all_val_index])]
                    eval_set = [(x_train.loc[self.all_val_index], y_train.loc[self.all_val_index])]
                    self.model.fit(x_train.loc[self.all_train_index], y_train.loc[self.all_train_index],
                                   eval_set=eval_set,
                                   verbose=self.verbose)
                else:
                    self.model.fit(x_train.loc[self.all_index], y_train.loc[self.all_index], verbose=self.verbose)


        elif self.param_opt == True:
            logger.info("\n")
            logger.info("Parameter optimization started.\n")

            refit = True

            if self.param_opt_method == "RandomizedSearchCV":
                self.model = RandomizedSearchCV(
                    estimator=self.model,
                    param_distributions=self.grid,
                    cv=self.cv,
                    n_iter=self.n_iter,
                    refit=refit,
                    scoring=scoring.get('scoring'),
                    n_jobs=self.nprocs,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    return_train_score=True
                )
            else:
                logger.info(
                    "RandomizedSearchCV optimization methods should be selected.\n")

            if self.train_size != None:
                if early_stop == 1:
                    eval_set = [(x_train.loc[self.smpl_val_index], y_train.loc[self.smpl_val_index])]
                    self.model.fit(x_train.loc[self.smpl_train_index], y_train.loc[self.smpl_train_index],
                                   eval_set=eval_set,
                                   verbose=self.verbose)

                    self.best_estimator = self.model.best_estimator_
                    self.cv_gini = 2 * (self.model.best_score_) - 1
                    self.model = self.best_estimator

                    if self.refit_to_all == True:

                        logger.info("Refitting to all data...")
                        eval_set = [(x_train.loc[self.all_val_index], y_train.loc[self.all_val_index])]
                        self.model.fit(x_train.loc[self.all_train_index], y_train.loc[self.all_train_index],
                                       eval_set=eval_set,
                                       verbose=self.verbose)

                    elif self.alpha != None:

                        eval_set = [(x_train.loc[self.smpl_val_index], y_train.loc[self.smpl_val_index])]
                        self.model.fit(x_train.loc[self.smpl_train_index], y_train.loc[self.smpl_train_index],
                                       eval_set=eval_set,
                                       verbose=self.verbose)

                elif early_stop == 0:
                    self.model.fit(x_train.loc[self.smpl_index], y_train.loc[self.smpl_index], verbose=self.verbose)

                    self.best_estimator = self.model.best_estimator_
                    self.cv_gini = 2 * (self.model.best_score_) - 1
                    self.model = self.best_estimator

                    if self.refit_to_all == True:

                        logger.info("Refitting to all data...")
                        self.model.fit(x_train.loc[self.all_index], y_train.loc[self.all_index], verbose=self.verbose)

                    elif self.alpha != None:
                        self.model.fit(x_train.loc[self.smpl_index], y_train.loc[self.smpl_index], verbose=self.verbose)

            else:

                if early_stop == 1:
                    eval_set = [(x_train.loc[self.all_val_index], y_train.loc[self.all_val_index])]
                    self.model.fit(x_train.loc[self.all_train_index], y_train.loc[self.all_train_index],
                                   eval_set=eval_set,
                                   verbose=self.verbose)

                    self.best_estimator = self.model.best_estimator_
                    self.cv_gini = 2 * (self.model.best_score_) - 1
                    self.model = self.best_estimator


                elif early_stop == 0:
                    self.model.fit(x_train.loc[self.all_index], y_train.loc[self.all_index], verbose=self.verbose)

                    self.best_estimator = self.model.best_estimator_
                    self.cv_gini = 2 * (self.model.best_score_) - 1
                    self.model = self.best_estimator

            logger.info("Results of %s" % self.best_estimator)

            logger.info("Mean CV score:  %s" % self.cv_gini)

        try:
            logger.info("Calculating SHAP feature importances...")
            feature_imp_df = get_feature_importances(self.model, x_train, x_train.sample(
                n=1000 if len(x_train.index) > 1000 else len(x_train.index), random_state=0),
                correlation_type=self.importance_direction_type, corr_df=self.targetcorr_df)

            # Putting it before plot incase exception happens there
            if os.getenv("MLFLOW_LOGGING"):
                feature_imp_df.to_csv("feature_imp.csv")
                mlflow.log_artifact("feature_imp.csv")
                os.remove("feature_imp.csv")

            fig = plot_feature_importances(feature_imp_df, n=self.n_feature_params)
            self.feature_importances = feature_imp_df
            if os.getenv("MLFLOW_LOGGING"):
                with open("feature_imp.png", "wb") as f:
                    fig.savefig(f)
                    mlflow.log_artifact("feature_imp.png")
                    os.remove("feature_imp.png")
        except Exception as e:
            logger.info("Feature importance calculation failed. %s" % e)
            self.feature_importances = None

        if early_stop:
            del eval_set

        cre_time = datetime.now()
        self.creation_ = str(cre_time)
        id_1 = str(round((self.start - datetime(1970, 1, 1)).total_seconds()))
        id_2 = str(round((cre_time - datetime(1970, 1, 1)).total_seconds()))
        rndm = str(random.randint(1000, 9999))
        self.id_ = id_1 + '-' + rndm + '-' + id_2
        self.train_time_ = (datetime.now() - self.start).total_seconds()

        logger.info("Model fitting completed.\n")

        self.train_prob = self.model.predict_proba(x_train)[:, 1]
        self.train_gini = 2 * roc_auc_score(y_train, self.train_prob) - 1
        logger.info("With ROC: " % roc_auc_score(y_train, self.train_prob))
        logger.info("With gini: " % (self.train_gini))

    def predict(self, x_test, y_test=[], sample='Validation'):
        logger = getLogger()
        x_test = x_test[self.model.feature_names_in_]
        self.test_pred_prob = self.model.predict_proba(x_test)[:, 1]
        self.test_pred_prob = np.round(self.test_pred_prob,38)

        test_pred = self.model.predict(x_test)

        sample_gini = None
        if len(y_test) >= 1:
            if os.getenv("MLFLOW_LOGGING"):
                eval_data = x_test.copy()
                eval_data['label'] = y_test.tolist()
                try:
                    model_uri = mlflow.get_artifact_uri('best_estimator')
                    mlflow.evaluate(
                        model_uri,
                        eval_data,
                        targets='label',
                        feature_names=x_test.columns.tolist(),
                        model_type="classification",
                        dataset_name="testset",

                    )
                except Exception as e:
                    logger.info(e)
            if sample == 'Validation':
                self.valid_gini = 2 * roc_auc_score(y_test, self.test_pred_prob) - 1
            elif sample == 'Test':
                self.test_gini = 2 * roc_auc_score(y_test, self.test_pred_prob) - 1
            else:
                sample_gini = 2 * roc_auc_score(y_test, self.test_pred_prob) - 1

            self.model_perf = {'Model Name': 'XGBoost', 'Phase': 'Base', 'Train Gini': self.train_gini,
                               'CV Gini': self.cv_gini, 'Valid Gini': self.valid_gini, 'Test Gini': self.test_gini}

            logger.info("Results of XGBoost")

            logger.info("Train gini: %s" % self.train_gini)
            logger.info("CV gini: %s" % self.cv_gini)
            logger.info("Valid gini: %s" % self.valid_gini)
            logger.info("Test gini: %s\n" % self.test_gini)
            if os.getenv("MLFLOW_LOGGING"):
                if self.train_gini:
                    mlflow.log_metric("train_gini", self.train_gini)
                if self.cv_gini:
                    mlflow.log_metric("cv_gini", self.cv_gini)
                if self.valid_gini:
                    mlflow.log_metric("valid_gini", self.valid_gini)
                if self.test_gini:
                    mlflow.log_metric("test_gini", self.test_gini)

        return self.test_pred_prob, test_pred

    def model_performance(self):
        logger = getLogger()
        logger.info("Results of XGBoost")

        logger.info("Train gini: %s" % self.train_gini)
        logger.info("CV gini: %s" % self.cv_gini)
        logger.info("Valid gini: %s" % self.valid_gini)
        logger.info("Test gini: %s\n" % self.test_gini)

        return self.model_perf



