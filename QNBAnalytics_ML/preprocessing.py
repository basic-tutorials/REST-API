import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from QNBAnalytics_ML.helpers import getLogger


def remove_features(df, features, inplace):
    df.drop(columns=features)
    return df
    
class conversions():

    def __init__(self):
        self.container = None
        
    def to_numeric(self, df, cols):
        logger = getLogger()
        self.to_numeric = cols
        
        try:
            df[cols].fillna(-987654321,inplace=True)
            df[cols] = df[cols].astype(int)
            df[cols].replace(-987654321,np.nan)
        except:
            try:
                df[cols].fillna(-987654321,inplace=True)
                df[cols] = df[cols].astype(float)
                df[cols].replace(-987654321,np.nan)
            except:
                logger.info("Couldn't convert to numeric.")
        return df

    def to_categoric(self, df, cols):
        logger = getLogger()
        self.to_categoric = cols
        try:
            df[cols] = df[cols].astype(str)
        except:
            logger.info("Couldn't convert to categoric.")
        return df    
        
class impute_null():
    
        def __init__(self, type='simple', strategy='mean', fill_value=None, add_indicator=False, n_feats=5):
            
            self.type = type
            self.strategy=strategy
            self.add_indicator=add_indicator
            self.fill_value = fill_value
            self.n_feats=n_feats
    
            if self.type=='simple':
                self.imp = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value, add_indicator=self.add_indicator)
            else:
                raise Exception("Imputation type is not valid")
    
        def fit(self, local_df):
            logger = getLogger()
            logger.info("Imputation fit started...")
            self.imp.fit(local_df)
            logger.info("Imputation fit completed...")
            
        
        def transform(self, local_df):
            logger = getLogger()
            logger.info("Imputation transform started...")
            new_local_df = self.imp.transform(local_df)
            logger.info("Imputation transform completed...")
            
            return new_local_df
            
        
        def fit_transform(self, local_df):
            logger = getLogger()
            logger.info("Imputation fit_transform started...")
            new_local_df = self.imp.fit_transform(local_df)    
            logger.info("Imputation fit_transform completed...")
            
            return new_local_df


class encode():
    
        def __init__(self, method = "Target Encoder", sample = 50000, min_samples_leaf=50):
            
            self.encoder = None
            self.cols_to_enc = None
            self.method = method
            self.sample = sample
            self.min_samples_leaf = min_samples_leaf
            
        def fit(self, x_train, y_train, cols_to_enc):
            logger = getLogger()
            logger.info("Encoding started...\n")
            
            if self.method == "Target Encoder":
                if self.sample == -1:
                    n = len(x_train)
                elif (self.sample>0) & (self.sample<=1):
                    n = int(self.sample*len(x_train))
                elif self.sample > 1:
                    n = min(self.sample,len(x_train))
                self.cols_to_enc = list(cols_to_enc)            
                self.encoder = TargetEncoder(min_samples_leaf=self.min_samples_leaf)  
                
                random.seed(48)
                sample_indices = random.sample(list(x_train.index), n)
                
                if len(sample_indices) > 0:
                    self.encoder.fit(x_train.loc[sample_indices, self.cols_to_enc] , y_train[sample_indices])
                    
                print("Encoding completed.\n")
                    
            else:
                logger.info("Select Target Encoder as Method.\n")
        
        def transform(self, df):
            logger = getLogger()
            logger.info("Encoding started...\n")
            
            df.loc[:,self.cols_to_enc] = self.encoder.transform(df.loc[:,self.cols_to_enc])
                    
            logger.info("Encoding completed.\n")
          
            return df

        def decoding(train, train_enc, woe_dict2, cols_to_enc):
    
            data_id=train.columns[0]
            for degisken in cols_to_enc:
                train_df=train[data_id].to_frame()
                train_df=pd.concat([train_df,train[degisken]],axis=1)
                train_df=pd.concat([train_df,train_enc[degisken]],axis=1)
                train_df.drop(columns=data_id, inplace=True)
                train_df=train_df.drop_duplicates()
                train_df=train_df.reset_index()
                train_df.drop(columns="index", inplace=True)
                df=pd.DataFrame(columns=['0','1','2','3','4'])
                for m in range(len(train_df)):
                    
                    for i in range(woe_dict2[degisken]["interval_end_exclude"].size):
                        if len(woe_dict2[degisken]["interval_end_exclude"])>1:
                            if woe_dict2[degisken]["interval_end_exclude"][0]>woe_dict2[degisken]["interval_end_exclude"][1]:
                                if (train_df.iloc[m,1] > woe_dict2[degisken]["interval_end_exclude"][i])  and (train_df.iloc[m,1] <= woe_dict2[degisken]["interval_start_include"][i]):
                                    df=df.append({i: train_df.iloc[m,0]},ignore_index=True)
                            else:
                                if (train_df.iloc[m,1] < woe_dict2[degisken]["interval_end_exclude"][i])  and (train_df.iloc[m,1] >= woe_dict2[degisken]["interval_start_include"][i]):
                                    df=df.append({i: train_df.iloc[m,0]},ignore_index=True)
                        else:
                            df=df.append({i: train_df.iloc[m,0]},ignore_index=True)
                for i in range(woe_dict2[degisken]["interval_end_exclude"].size):    
                    #df.replace(np.nan, "", inplace=True)
                    deger=''
                    for value in df[i].unique():
                        if str(value)!= "nan":
                            if str(value)!='' and deger!='':
                                deger=deger+', '+str(value)
                            else:
                                deger=deger+str(value)
                    woe_dict2[degisken]["interval_end_exclude"][i]=''
                    woe_dict2[degisken]["interval_start_include"][i]=deger
            
            return woe_dict2
        
        def decode(self, x_train, train_enc, woe_dict):
            logger = getLogger()
            logger.info("Decoding started...\n")
            
            #self.decode=copy.deepcopy(woe_dict)
            self.decode=self.encoder.decoding(x_train,train_enc,self.decoder,self.cols_to_enc)	
            
            logger.info("Decoding completed.\n")

class scale():
    
        def __init__(self, type='StandardScaler', copy=True, with_mean=True, with_std=True, feature_range=(0, 1)):
            self.type = type
            self.copy = copy
            self.with_mean = with_mean
            self.with_std = with_std
            self.feature_range = feature_range

            if self.type == 'StandardScaler':
                self.scaler = StandardScaler(copy=self.copy, with_mean=self.with_mean, with_std=self.with_std)
            else:
                raise Exception("Scaling type is not valid")
            
        def fit(self,local_df):
            logger = getLogger()
            logger.info("Scaling fit started...\n")
            self.scaler.fit(local_df) 
            logger.info("Scaling fit completed.\n")
            
        def transform(self, local_df):
            logger = getLogger()
            logger.info("Scaling started...\n")
            local_df2 = self.scaler.transform(local_df)     
            logger.info("Scaling completed.\n")
          
            return local_df2
        
        def fit_transform(self,local_df):
            logger = getLogger()
            logger.info("Scaling started...\n")
            local_df2 = self.scaler.fit_transform(local_df) 
            logger.info("Scaling completed.\n")
        
            return local_df2
