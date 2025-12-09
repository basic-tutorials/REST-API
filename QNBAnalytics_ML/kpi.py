# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from scipy.stats import ks_2samp

class ClassificationScores():

    def roc_auc_score(y_true, y_pred_prob):
        roc_score=roc_auc_score(y_true, y_pred_prob)
        print(print("\n Roc Auc Score: ",roc_score))
        return roc_score
    
    def gini_score(y_true, y_pred_prob):
        gini=2*roc_auc_score(y_true, y_pred_prob)-1
        print("\n Gini: ",gini)
        return gini

    def confusion_matrix(y_true,y_pred):
        CM=confusion_matrix(y_true,y_pred)
        TN=CM[0][0]
        FN=CM[1][0]
        TP=CM[1][1]
        FP=CM[0][1] 
        print("/n Confusion Matrix: ",CM)
        return CM
        
    def lift_score(y_true, y_pred):
        CM=confusion_matrix(y_true,y_pred)
        TN=CM[0][0]
        FN=CM[1][0]
        TP=CM[1][1]
        FP=CM[0][1]           
        lift_score= (TP/(TP+FP))/((TP+FN)/(TP+TN+FP+FN))
        print("\n Lift score: ",(lift_score))
        return lift_score

    def ks_statistic(predict_prob, y_true):
        ks_statistic=ks_2samp(predict_prob[y_true==0].to_numpy().ravel(), predict_prob[y_true==1].to_numpy().ravel())
        print("\n KS Statistics: ",ks_statistic)
        return ks_statistic.statistic      

    
    def lift_ks_tables(y_test, y_pred):
           
        data=pd.concat([y_test.reset_index(),pd.DataFrame(data=y_pred, columns=["predict"])], axis=1) 
        data= data.sort_values(by="predict",ascending=False)
        data["row_id"] = range(0,0+len(data))
        data["decile"] = ( data["row_id"] / (len(data)/10) ).astype(int)
        # Check the count by decile
        data.loc[data["decile"] == 10]=9
        data["decile"].value_counts()
        #create gains table
        gains = data.groupby("decile")[y_test.name].agg(["count","sum"])
        gains.columns = ["count","actual"]
        gains
        #add metrics to the gains table
        gains["non_actual"] = gains["count"] - gains["actual"]
        gains["cum_count"] = gains["count"].cumsum()
        gains["cum_actual"] = gains["actual"].cumsum()
        gains["cum_non_actual"] = gains["non_actual"].cumsum()
        gains["percent_cum_actual"] = (gains["cum_actual"] / np.max(gains["cum_actual"])).round(2)
        gains["percent_cum_non_actual"] = (gains["cum_non_actual"] / np.max(gains["cum_non_actual"])).round(2)
        gains["if_random"] = np.max(gains["cum_actual"]) /10 
        gains["if_random"] = gains["if_random"].cumsum()
        gains["lift"] = (gains["cum_actual"] / gains["if_random"]).round(2)
        gains["K_S"] = np.abs( gains["percent_cum_actual"] - gains["percent_cum_non_actual"] ) * 100
        gains["gain"]=(gains["cum_actual"]/gains["cum_count"]*100).round(2)
        gains = pd.DataFrame(gains)
         
        return gains

    
    def all_class_scores(y_true, y_pred, y_pred_prob):
        roc_score=ClassificationScores.roc_auc_score(y_true,y_pred_prob)
        gini=ClassificationScores.gini_score(y_true, y_pred_prob)
        cm=confusion_matrix(y_true,y_pred)
        lift_score=ClassificationScores.lift_score(y_true, y_pred)
        ks_statistic=ClassificationScores.ks_statistic(y_pred_prob, y_true)
        scores = {'roc_score': roc_score,
                  'gini': gini, 
                  'cm': cm, 
                  'lift_score': lift_score, 
                  'ks_statistic': ks_statistic}
        
        return scores
