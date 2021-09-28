from typing import List
from pandas_profiling import ProfileReport
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def EDA(df: pd.DataFrame, title: str, name_of_file: str): 
    fraud=df[df['fraud']==1]
    not_fraud=df[df['fraud']==0]
    profile =ProfileReport(df, title=title, explorative= True)
    profile.to_file(f"{name_of_file}.html")
    return (fraud, not_fraud)

def vif_factor(df: pd.DataFrame, columns: list):
    vif_data = pd.DataFrame()
    X= df[columns]
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data

def redifine_df(df: pd.DataFrame, columns: list):
    return df[columns]

def under_sample( df: pd.DataFrame, class_variable: str , sample_size: int, sample_class: int):
    positive_df= df[df[class_variable]==1]
    negative_df= df[df[class_variable]==0]

    if sample_class==1:
        positive_df= positive_df.sample(sample_size)
    else:
        negative_df= negative_df.sample(sample_size)
    
    frames=[positive_df, negative_df]
    df=pd.concat(frames, axis=0).reset_index(drop= True)
    return df

def split_test_train(df: pd.DataFrame, class_variable: str,columns: list, test_size: int ):
    X = df[columns].values
    y= df[class_variable].values
    return (train_test_split(X, y, test_size = test_size, random_state = 0))

def smote(X_train: np.ndarray, y_train: np.ndarray):
    sm = SMOTETomek()
    return sm.fit_resample(X_train, y_train)

def feature_scaling(X_train: np.ndarray, X_test: np.ndarray):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return (X_train, X_test)