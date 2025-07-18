import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models,plot_qq

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression":LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(),
                "CatBoost": CatBoostClassifier(verbose=False),
                "AdaBoost": AdaBoostClassifier(),
                "K-Nearest Neighbors": KNeighborsClassifier()                 
 
            }

            params = {
                "Logistic Regression": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ['liblinear', 'lbfgs'],
                    "penalty": ['l2']
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 150],
                    "max_depth":  [None,10,20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                   
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth":  [5,10,None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0]
                },
                "XGBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 6],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0]
                },
                "CatBoost": {
                    "depth": [6, 8],
                    "iterations": [100, 200],
                    "learning_rate": [0.03, 0.1],
                    "l2_leaf_reg": [1, 3, 5, 7]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1]
                },
                "K-Nearest Neighbors": {
                    "n_neighbors": [3, 5, 7]
                    
                }
            }

            model_report,best_model,best_model_name = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )
            logging.info(f"Model evaluation report: {model_report}")
           
            best_model_score=model_report[best_model_name]
            
            if best_model_score < 0.6:
                    
             logging.warning(f"No sufficiently accurate model found. Best score: {best_model_score}")

            else:
                logging.info(f"Best model: {best_model_name} with score: {best_model_score}")


            # if best_model_score < 0.6:
            #     raise Exception("No best model found")

            # logging.info(f"Best model found: {best_model_name} with accuracy: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            final_accuracy = accuracy_score(y_test, predicted)
        
            # ✅ Plot QQ plot of residuals
            residuals = y_test - predicted
            qqplot_path = os.path.join("artifacts", "qqplot_residuals.png")
            plot_qq(
                residuals,
                title="QQ Plot - Residuals",
                save_path=qqplot_path
            )
            logging.info(f"Residual QQ plot saved at: {qqplot_path}")

            return final_accuracy

          

        except Exception as e:
            raise CustomException(e, sys)
