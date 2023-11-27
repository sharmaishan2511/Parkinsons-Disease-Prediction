import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,cross_val_score

from src.exception import CustomException

def select_lasso(X_train,y_train):
    try:
        feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
        feature_sel_model.fit(X_train, y_train)
        selected_feat = X_train.columns[(feature_sel_model.get_support())]
        print(selected_feat)
        return selected_feat
    except Exception as e:
        return (e,sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}
        kf = KFold(10)

        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            crossval = cross_val_score(model,X_train,y_train,cv=kf)

            report[list(models.keys())[i]] = [train_model_score,test_model_score]

        return report

    except Exception as e:
        raise CustomException(e, sys)