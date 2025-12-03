import inspect
import warnings
import sys

#import shap
import pandas as pd
import numpy as np
from sklearn.base import is_classifier
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
np.random.seed(48)


def _mean_shap_categoric_var(shap_values_df, input_data_df, cat_cols):
    cats_dfs = []

    for col in cat_cols:
        a, b = shap_values_df[col], input_data_df[col]
        a.name, b.name = "shap", "category"

        cat_df = pd.concat([a, b], axis=1)
        cat_df = cat_df.groupby(by=["category"], as_index=False)["shap"].mean().sort_values(by="shap", ignore_index=True)
        cat_df["column"] = col
        
        cats_dfs.append(cat_df)
    
    if cats_dfs:
        cats_df = pd.concat(cats_dfs, ignore_index=True)
    else:
        cats_df = pd.DataFrame()

    return cats_df

def _shap_correlation(shap_values_df, input_data_df, feature_names, cat_cols):
    # Calculate correlation
    corr_list = []
    for col in feature_names:
        if col in cat_cols:
            corr_list.append(np.nan)
            continue
            
        a, b = shap_values_df[col], input_data_df[col]
        corr = a.corr(b)
        corr_list.append(corr)

    return corr_list

    
def _raw_target_correlation(df, corr_df):
    return df.join(corr_df)["correlation"].tolist()

def _get_shap_feature_importances(shap_values, X_test, correlation_type="shap", corr_df: pd.DataFrame = None) -> pd.DataFrame:

    if correlation_type == "factor":
        assert corr_df is not None, "Corr_df must be given."

    feature_names = X_test.columns
    cat_cols = X_test.select_dtypes("category").columns

    abs_shap_vals = np.abs(shap_values).mean(0)
    df = pd.DataFrame(list(zip(feature_names, abs_shap_vals)),  columns=['variable', 'abs_shap']).set_index("variable")
    df["feature_importance"] = (df['abs_shap'] / sum(df['abs_shap'])*100)

    # Calculate correlations between shap values and input data.
    shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
    input_data_df = X_test.reset_index(drop=True)

    if correlation_type == "shap":
        corr_list = _shap_correlation(shap_values_df, input_data_df, feature_names, cat_cols)
        
    elif correlation_type == "factor":
        corr_list = _raw_target_correlation(df, corr_df)

    df["correlation"] = corr_list
    df = df[["feature_importance", "correlation"]]
    df = df.sort_values(by=['feature_importance'], ascending=False)

    cats_df = _mean_shap_categoric_var(shap_values_df, input_data_df, cat_cols)

    return df, cats_df



def get_feature_importances(model, X_train: pd.DataFrame, X_test: pd.DataFrame, agnostic = False, fast = True,
    correlation_type:str = "shap", corr_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Get feature importance from given model.
    
    Args:
        model: Sklearn/TF/Pytorch/XGBoost/CatBoost/LightGBM model or prediction probability function if agnostic option is enabled.
        X_train: Training set for SHAP.
        X_test: Test set for SHAP.
        agnostic: Option for enabling model agnostic SHAP calculations.
        fast: Option for enabling fast but slightly less accurate SHAP calculations.
    
    Returns:
        df: Feature importance dataframe.
    
    
    *** Important Notes for Use Cases ***
        - X_train and X_test must be fully preprocessed.
        
        - Using small X_train and X_test is recommended. Especially for X_test. Usually 1k is more than enough.
        
        - Fast setting achieves speed with these considerations:
              Agnostic Model ---> Median values for training set is used.
              Torch/TF Model ---> Training set is not used.
              Tree     Model ---> Training set is not used.
              Linear   Model ---> Fast option is not available.
                    
        - If agnostic model is preferred, user must provide a probability function as a model. 
        
        For example:
            KNeighborsClassifier from scikit-learn must be provided as 
            >>> knn = KNeighborsClassifier()
            >>> get_feature_importances(lambda x: knn.predict_proba(x)[:,1], 
                        X_train.sample(n=100, random_state=1).fillna(0), 
                        X_test.sample(n=100, random_state=422).fillna(0),
                        agnostic=True,
                        fast=False
                       )
    
    
    """
    features = X_train.columns
    
    # Check if model is linear. If so set fast=False because SHAP needs traning set for linear models.
    linear_models = inspect.getmembers(sys.modules["sklearn.linear_model"], inspect.isclass)
    linear_clfs = [clf[1] for clf in linear_models if is_classifier(clf[1])]
    if any([isinstance(model, clf) for clf in linear_clfs]):
        warnings.warn("Model is linear, disabling fast option...") if fast else print("")
        fast=False
    
    # Give warnings for non agnostic models.
    if not agnostic:
        warnings.warn(f"""Your model is {model}""")
        warnings.warn(f"""If your model is not TreeBased, Linear or Deep Learning model, make sure to use agnostic option and give model as predict_proba function.""")
    
    if correlation_type not in ["shap", "factor"]:
        raise ValueError("Correlation_type is not valid. Valid: `shap` and `factor`")
    

    # Main
    if agnostic and fast:
        med = (X_train.median().values).reshape((1,X_train.shape[1]))
        med = pd.DataFrame(med, columns=features)
        
        explainer = shap.explainers.Permutation(model, med, max_evals = 2001)
        # This fails because of a bug. https://github.com/slundberg/shap/issues/1526
        # shap_values = explainer.shap_values(X_test)
        shap_values = explainer(X_test).values
        shap_values = shap_values[:,:,1]
            
    elif agnostic and not fast:
        explainer = shap.explainers.Permutation(model, X_train, max_evals = 2001)
        # This fails because of a bug. https://github.com/slundberg/shap/issues/1526
        # shap_values = explainer.shap_values(X_test)
        shap_values = explainer(X_test).values
        shap_values = shap_values[:,:,1]

        
    elif not agnostic and fast:
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(X_test)

            
    elif not agnostic and not fast:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer.shap_values(X_test)


    # Fix shap_values if needed.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    df, cats_df = _get_shap_feature_importances(shap_values, X_test, correlation_type=correlation_type, corr_df = corr_df)

    return df


def plot_feature_importances(df, n=50, save_figure=False, name_ext=""):
    """
    Args:
        df: Feature importance dataframe including shap and abs_shap_normalized values.
        n: Number of top features to plot.

    Returns:
        matplotlib.pyplot.figure
    """

    gdf = df.iloc[:n,:].copy()

    fig = plt.figure(figsize=(12,8))
    x = gdf["feature_importance"].values
    y = gdf.index.values

    colors = []
    for corr in gdf["correlation"]:
        if corr > 0:
            colors.append("deepskyblue")
        elif corr < 0:
            colors.append("red")
        else: # nans and 0's
            colors.append("black")

    sns.barplot(x=x, y=y, palette=colors)

    plt.title("Feature Importances", fontsize=15)
    plt.xlabel("Relative Feature Importance in Percentages", fontsize=12)
    plt.plot([],[], color='deepskyblue', label="Positive Correlation", linewidth=5.0)
    plt.plot([],[], color='red', label="Negative Correlation", linewidth=5.0)
    plt.plot([],[], color='black', label="No Correlation", linewidth=5.0)
    plt.legend()
    
    plt.tight_layout()
    
    if save_figure==True:
        plt.savefig(f'Figures/Feature_Importance_{name_ext}.png')

    return fig
