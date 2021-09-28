from V1 import X_train
import pandas as pd
import numpy as np
import xgboost
from preprocessing import *
from models import *
from evaluation import *

class FrauDetectionModel:
    def __init__(self, df, columns: list, class_variable: str):
        self.columns = columns
        self.df= df[columns]
        self.class_varible = class_variable,
    
    ############## Preprocessing##############
    def EDA(self, title: str, name_of_file: str): 
       return EDA(self.df, title, name_of_file)

    def vif_factor(self, columns: list):
        return vif_factor(self.df, columns)

    def redifine_df(self, columns: list):
        self.columns= columns
        self.df = redifine_df(self.df, columns)

    def under_sample(self, class_variable:str , sample_size: int, sample_class: int):
        self.df = under_sample(self.df, class_variable, sample_size, sample_class )

    def test_train_split(self, class_variable, columns, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = split_test_train(self.df, class_variable, columns, test_size)

    def smote(self):
        self.X_train, self.y_train = smote(self.X_train, self.y_train)

    def feature_scaling(self):
       feature_scaling(self.X_train, self.X_test)

    
    ################ Model Selection and Training################
    # SVM Classifier
    def svm(self, kernel: str, gamma: int):
       self.svm = svm_classifier(self.X_train, self.y_train, kernel, gamma)

    #Random Forest Classifier
    def random_forest(self, n_estimators: int, criterion: str ):
        self.random_forest_classifier = random_forest_classifier(self.X_train, self.X_test, n_estimators, criterion)

    #XGBoost Classifier
    def xgboost(self):
        self.xgboost_classifier = xgboost(self.X_train, self.y_train)

    #ANN classifier
    def ann(self, metrics: list, optimizer: str, epochs: int, batch_size: int ):
        self.ann_classifier = ann_classifier(self.X_train, self.y_train, metrics, optimizer, epochs, batch_size )
        

    ############### Pridicting results and Evaluating Performance#############

    def predict_results(self, classifier , ann: bool):
       self.y_pred = predict_results(classifier, self.X_test, bool )

    def grid_search(self, parameters: list(dict), classifier, scoring: list, k_value: int, X_train: np.ndarray, y_train: np.ndarray):
        grid_search(parameters, classifier, scoring, k_value, self.X_train, self.y_train)

    def k_fold_cv(self, classifier, k_value):
        k_fold_cv(k_value, self.X_train, self.y_train)
    
    def model_evaluation(self, classifier):
        model_evaluation(classifier, self.y_test, self.y_pred, self.X_test)
