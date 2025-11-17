# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from QNBAnalytics_ML.helpers import getLogger
#import cx_Oracle

#from sqlalchemy import types
#from sqlalchemy.engine import create_engine

try:
    cx_Oracle.init_oracle_client(lib_dir=os.getenv("cx_oracle_client_path"))
except:
    pass

def connect_to_sql(username='username', password='password'):
    #logger = getLogger()
    #DIALECT = 'oracle'
    #SQL_DRIVER = 'cx_oracle'
    #USERNAME = username #enter your username
    #PASSWORD = password #enter your password
    #HOST =  	#enter the oracle db host url
    #PORT =  	#enter the oracle port number
    #SERVICE =  	#enter the oracle db service name
    #ENGINE_PATH_WIN_AUTH = DIALECT + '+' + SQL_DRIVER + '://' + USERNAME + ':' + PASSWORD +'@' + HOST + ':' + str(PORT) + '/?service_name=' + SERVICE
    #engine = create_engine(ENGINE_PATH_WIN_AUTH)
    try:
        engine.execute('select 1 from dual')
        logger.info("Connected!\n")
    except Exception as e:
        raise e
        logger.info("Connection error! %s" % e)
    
    return engine

def data_load(engine = None, sql = None, data=None, index=None, encoding='utf-8', chunksize=25000):
    
        logger = getLogger()
        if engine != None: #if sql is used

            try:
                if sql:
                    execute = sql
                else:
                    execute = "select * from "+data

                dfx = []
                df = pd.DataFrame()
                for chunk in pd.read_sql_query(execute, engine, chunksize=chunksize):
                    dfx.append(chunk)
                df = pd.concat(dfx, ignore_index=True)
                df.columns = map(str.upper, list(df.columns))
                del dfx

                if index:
                    df = df.set_index(index)

                logger.info("Dataframe created!\n")
                logger.info(df.shape)
                return df
            except Exception as e:
                logger.info("Check your sql connection or data (table name)\n")
                logger.info("Data Load Error! %s" % e)
                
        else: #if local csv is used
            
            if data != None:
                try:
                    extension = (data[data.rindex('.')+1:] if data != None else None)
                    if extension == 'csv':
                        dfx = []
                        df = pd.DataFrame()
                        for chunk in pd.read_csv(data,encoding=encoding, chunksize=chunksize):
                            dfx.append(chunk)
                        df = pd.concat(dfx, ignore_index=True)
                        df.columns = map(str.upper, list(df.columns))
                        if index:
                            df = df.set_index(index)
                        print("Dataframe created!\n")
                        logger.info("Dataframe created!\n")

                    elif extension == 'xls':
                        dfx = []
                        df = pd.DataFrame()
                        for chunk in pd.read_excel(data, chunksize=chunksize):
                            dfx.append(chunk)
                        df = pd.concat(dfx, ignore_index=True)
                        df.columns = map(str.upper, list(df.columns))
                        if index:
                            df = df.set_index(index)
                        print("Dataframe created!\n")
                        logger.info("Dataframe created!\n")

                    elif extension == 'xlsx':
                        dfx = []
                        df = pd.DataFrame()
                        for chunk in pd.read_excel(data, chunksize=chunksize):
                            dfx.append(chunk)
                        df = pd.concat(dfx, ignore_index=True)
                        df.columns = map(str.upper, list(df.columns))
                        if index:
                            df = df.set_index(index)
                        print("Dataframe created!\n")
                        logger.info("Dataframe created!\n")
                    print(df.shape)
                    return df

                except Exception as e:
                    logger.info("Check your data path\n")
                    logger.info("Data Load Error! %s" % e)
    
def train_test(df, test_size = None, random_state=1, split_style ='random', group = None, column_name = None, sample = None,  stratify = None):
        logger = getLogger()    
        logger.info("Train/Test Split Started")
        if test_size<1:
            test_size=round(len(df)*test_size)
        if split_style =='random':
            train, test = train_test_split(df,test_size=test_size,random_state=random_state,stratify=stratify)
        else:
            gs = GroupShuffleSplit(n_splits=2, train_size=1, random_state=48)
            train_ix, test_ix = next(gs.split(df, groups=df[group]))
            train = df.loc[train_ix]         
            test = df.loc[test_ix]
            
        logger.info("Train/Test Split Completed")
        #traintest_used = True
        return train, test

def explore(local_df):
        logger = getLogger()
        no_min = False
        
        try:
            stats = local_df.describe(include="all").transpose()[["count","mean","min","max","std"]]
        except Exception as e:
            logger.info("min/max calculation failed. %s" % e)
            stats = local_df.describe(include="all").transpose()[["count"]]
            no_min = True
    
        categoric_types = ["object","str", 'category']
        numeric_types = ["bool","int","float","complex","float32","float64","int32","int64","Int32","Int64"]
        date_types = ["datetime64[ns]"]
        
        datatypes = local_df.dtypes
        
        for dt in categoric_types:
            datatypes = datatypes.replace(dt,"categoric")
        for dt in numeric_types:
            datatypes = datatypes.replace(dt,"numeric")
        for dt in date_types:
            datatypes = datatypes.replace(dt,"date")
            
            
        missing_rate = local_df.isnull().sum() * 100 / len(local_df)
        unique_count = local_df.nunique()
        additional_stats = pd.DataFrame({
                                        'datatypes': datatypes, 
                                        'missing_rate': missing_rate, 
                                        'unique_count': unique_count
                                    })
    
        summary = pd.merge(stats, additional_stats, left_index = True, right_index = True)
    
        summary.loc[:,"role"] = "input"
    
        if len(summary[summary["count"] == summary["unique_count"]]) == 1:
            summary.loc[summary["count"] == summary["unique_count"],"role"] = 'ID'
            logger.info("Unique ID detected!")
        elif len(summary[summary["count"] == summary["unique_count"]]) > 1:
            summary.loc[summary["count"] == summary["unique_count"],"role"] = 'ID'
            logger.info("There might be multiple ID in dataframe")
        else:
            logger.info("No ID detected!")
    
        summary.loc[summary.index.isin(["TARGET","target","Target"]),"role"] = 'target'

        if no_min:
            summary = summary[["datatypes","role","count","unique_count","missing_rate"]]
        else:
            summary = summary[["datatypes", "role", "count", "unique_count", "missing_rate", "mean", "min", "max", "std"]]
        
        summary.loc[:,"use"] = True

        return summary
        
def export_to_sql(engine, df, username, password, table_name):

        table_name = table_name.lower()  
        table = table_name 
        connection = engine.connect()
        
        dtype = {c:types.VARCHAR(df[c].str.len().max()) 
                for c in df.columns[df.dtypes == 'object'].tolist()}
        
        df.to_sql(table, connection, dtype=dtype, if_exists='replace', chunksize = 10**4, index=False)
        

def sample(df, n=None, random_state=None):
    fraction = False
    if n is None:
        fraction = True
        n = 0.1  # default value
    elif 0 < n < 1:
        fraction = True
    elif 1 <= n <= len(df):
        pass
    else:
        raise Exception("Sample size can not be greater than the number of rows.")

    if fraction:
        sampled_df = df.sample(frac=n, random_state=random_state)
    else:
        sampled_df = df.sample(n=n, random_state=random_state)

    return sampled_df


def merge(left, right, how="inner", on=None):
    # convert series to DataFrame
    if type(left) is pd.Series:
        left = pd.DataFrame(left)

    if type(right) is pd.Series:
        right = pd.DataFrame(right)

    if type(left) is pd.DataFrame and type(right) is pd.DataFrame:
        if on is None:
            merged_df = left.merge(right, how=how, left_index=True, right_index=True)
        else:
            merged_df = left.merge(right, how=how, on=on, suffixes=(None, '_y'))

    elif type(left) is pd.DataFrame and type(right) is np.ndarray:
        merged_df = left.copy()
        merged_df["merged_col"] = right

    elif type(left) is np.ndarray and type(right) is pd.DataFrame:
        merged_df = left.copy()
        merged_df["merged_col"] = left

    elif type(left) is np.ndarray and type(right) is np.ndarray:
        right_df = pd.DataFrame(right)
        left_df = pd.DataFrame(left)
        merged_df = right_df.merge(left_df, left_index=True, right_index=True)

    else:
        raise Exception(f"Merge operation is not supported between {type(left)} and {type(right)}")

    return merged_df

def conversion(num_variables, cat_variables, user_cats=None, user_nums=None):

    logger = getLogger()
    logger.info("Data Type Conversion started")
    if user_cats is not None:

        if num_variables is not None and not num_variables.empty:
            num_variables = num_variables.drop(user_cats, errors='ignore')

        if cat_variables is not None and not cat_variables.empty:
            cat_variables = cat_variables.union(user_cats)

    if user_nums is not None:
        if cat_variables is not None and not cat_variables.empty:
            cat_variables = cat_variables.drop(user_nums, errors='ignore')
        if num_variables is not None and not num_variables.empty:
            num_variables = num_variables.union(user_nums)

    logger.info("Data Type Conversion completed")
    return num_variables, cat_variables
