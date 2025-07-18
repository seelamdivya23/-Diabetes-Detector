import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=r"D:\DATA SCIENCE\ML-PROJECTS\Diabetes-Detector\src\components\artifacts\model.pkl"
            preprocessor_path=r'D:\DATA SCIENCE\ML-PROJECTS\Diabetes-Detector\src\components\artifacts\preprocessor.pkl'
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Pregnancies:int,
        Glucose:int,
        BloodPressure:int,
        SkinThickness:int,
        Insulin:int,
        BMI:int,
        DiabetesPedigreeFunction:int,
        Age:int):
       
        self. Pregnancies =  Pregnancies
        self.Glucose = Glucose
        self. BloodPressure =  BloodPressure
        self.SkinThickness =SkinThickness
        self. Insulin =  Insulin
        self. BMI =  BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self. Age = Age

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Pregnancies": [self.Pregnancies],
                "Glucose": [self. Glucose],
                "BloodPressure": [self. BloodPressure],
                "SkinThickness": [self.SkinThickness],
                "Insulin": [self.Insulin],
                "BMI": [self. BMI],
                "DiabetesPedigreeFunction": [self.DiabetesPedigreeFunction],
                "Age": [self. Age],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

