grids = {

    "logistic_regression" : {
                'penalty': ['l2'],
                'solver': ['lbfgs'],
                'C': [0.1, 0.3, 0.5, 1],
                'max_iter': [100]
            },

    "random_forest_classifier" : {
                # 'class_weight': ['balanced',None],
                'max_depth': [3, 5, 8, 10, 12],
                'max_features': ['sqrt'],
                'min_samples_leaf': [5, 20, 100, 300, 500],
                'n_estimators': [100]
            },

    "lightgbm_classifier" : {
                'n_estimators': [5000],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [7, 15, 30],
                'max_depth': [-1, 3, 5],
                'min_child_samples': [50, 500],
                'early_stopping_round': [10],
                'first_metric_only': [True],
                'n_jobs': [1],
                'reg_alpha': [0],
                'reg_lambda': [0],
                'min_split_gain': [0, 2, 4]
            },

    "xgboost_classifier" : {
                        'learning_rate': [0.1, 0.3],
                        'max_depth': [3, 6],
                        'min_child_weight': [50, 300, 5000],
                        'lambda': [1],
                        'alpha': [0],
                        'gamma': [0,5],
                        'subsample': [1],
                        'colsample_bytree': [1],
                        'n_estimators': [1000],
                        'early_stopping_rounds': [10],
                        'eval_metric': ["auc"],
                        'tree_method': ['hist']
            }
}