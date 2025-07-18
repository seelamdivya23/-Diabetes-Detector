import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}
        best_model_obj = None
        best_model_name = None
        best_score = 0
        for name, model in models.items():
            params = param[name]
            gs = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
            gs.fit(X_train,y_train)

            # model.set_params(**gs.best_params_)
            # model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model
            y_test_pred = gs.predict(X_test)
            test_model_score=accuracy_score(y_test,y_test_pred)
            report[name]=test_model_score

            if test_model_score>best_score:
                best_score=test_model_score
                best_model_obj=gs.best_estimator_
                best_model_name=name

        return report,best_model_obj,best_model_name

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
# ✅ New function: QQ plot for normality check
def plot_qq(data, title="QQ Plot", save_path=None):
    """
    Plots a QQ plot for the given data and optionally saves it.
    """
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    plt.figure(figsize=(6, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(title)

    if save_path:
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path)

    plt.close()
