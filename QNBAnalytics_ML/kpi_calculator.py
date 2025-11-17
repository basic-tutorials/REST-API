# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


def kpi_calculator(varlist, x_train, y_train):
    # Calculate ginis for each selected variable
    ginis = []
    rocs = []
    for var in varlist:
        roc=round(roc_auc_score(y_train, x_train[var]),4)
        gini = round(abs(2*roc-1)*100,2)
        ginis.append(gini)
        rocs.append(roc)
    cluster_df = pd.DataFrame({'Variables': varlist, 'Gini': ginis, 'ROC': rocs})
    return cluster_df


def cv_score(estimator_, x_train, y_train):
    # Calculate ginis for each selected variable
    print("Results of ", estimator_)
    print()
    print("===Cross-val Scores:=== \n")
    cv_scores = cross_val_score(estimator_, x_train, y_train,cv=5,scoring='roc_auc')
    print(cv_scores)
    print("Average CV Score is: ",cv_scores.mean())
    print()
    print("Average CV gini is ",(2*cv_scores.mean()-1))
    print()
    return cv_scores