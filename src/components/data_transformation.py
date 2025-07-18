# Import system and dataclass utilities
import sys
from dataclasses import dataclass

# Add project root to the system path to resolve 'src' module imports
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import required libraries
import numpy as np                        # For numerical operations
import pandas as pd                       # For reading and handling datasets

# Scikit-learn modules for preprocessing
from sklearn.compose import ColumnTransformer             # For applying pipelines to column subsets
from sklearn.impute import SimpleImputer                  # For handling missing values
from sklearn.pipeline import Pipeline                     # For chaining preprocessing steps
from sklearn.preprocessing import PowerTransformer         # For encoding and scaling

# Import custom components
from src.exception import CustomException                 # For custom error handling
from src.logger import logging                            # For logging execution steps
from src.utils import save_object                         # For saving objects like transformers
from src.utils import save_object, plot_qq                # ✅ Added plot_qq
# Import SMOTE
from imblearn.over_sampling import SMOTE

# # ✅ Top-level function to make values strictly positive (required for Box-Cox)
# def shift_positive(X):
#     X = np.array(X)
#     shift = np.abs(np.min(X, axis=0)) + 1e-4  # Small buffer to avoid zero
#     return X + shift

# Configuration class for storing transformation file path
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")  # Path to save preprocessing object

# Main class for data transformation
class DataTransformation:
    def __init__(self):
        # Load configuration settings for this component
        self.data_transformation_config = DataTransformationConfig()

    # Method to create and return the preprocessing object (a ColumnTransformer)
    def get_data_transformer_object(self):
        '''
        This function is responsible for setting up the transformation pipelines
        for both numerical and categorical columns.
        '''
        try:
            # Define the columns to apply transformations on
            numerical_columns = [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ]
            
            # shift_transformer = FunctionTransformer(shift_positive)

            # Pipeline for numerical columns: fill missing values and scale
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),      # Fill missing with median
                    ("yeo_johnson", PowerTransformer(method='yeo-johnson'))                        # Standardize (mean=0, std=1)
                    # ("shift", shift_transformer),  # ✅ ensure positivity
                    # ("box_cox", PowerTransformer(method='box-cox'))  # ✅ apply Box-Cox
                ]
            )

            # Log column information
            logging.info(f"Numerical columns: {numerical_columns}")

            # Create a ColumnTransformer to apply the above pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                ]
            )

            return preprocessor  # Return the configured transformer

        except Exception as e:
            raise CustomException(e, sys)  # Raise a custom exception if anything fails

    # Method to apply the transformations on training and test datasets
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load the train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read train and test complted")
               # Handle outliers using IQR method before transformation
            def remove_outliers_iqr(df, columns):
                for col in columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                return df
            logging.info("remove outliers")

            numerical_columns = [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ]

            train_df = remove_outliers_iqr(train_df, numerical_columns)
            test_df = remove_outliers_iqr(test_df, numerical_columns)
            # Define the target/output column
            target_column_name = "Outcome"

            # Separate input features and target for train set
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target for test set
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # Get the preprocessor object (ColumnTransformer)
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Fit and transform the training input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            # Only transform (not fit) the testing input features
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # ✅ Plot QQ plots for the first few features after transformation
            for i in range(min(3, input_feature_train_arr.shape[1])):  # plot first 3 features
                feature_data = input_feature_train_arr[:, i]
                plot_path = os.path.join("artifacts", f"qqplot_feature_{i}.png")
                plot_qq(
                    feature_data,
                    title=f"QQ Plot - Feature {i}",
                    save_path=plot_path
                )
                logging.info(f"QQ plot saved: {plot_path}")

            # # --- SMOTE Integration ---
            logging.info("Applying SMOTE on the training data.")
            smote = SMOTE(random_state=42)
            
            # # Apply SMOTE to the transformed training features and target
            input_feature_train_resampled, target_feature_train_resampled = smote.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            logging.info(f"Original training data shape: {input_feature_train_arr.shape}, {target_feature_train_df.shape}")
            logging.info(f"Resampled training data shape: {input_feature_train_resampled.shape}, {target_feature_train_resampled.shape}")
            # --- End SMOTE Integration ---
        
            # Concatenate input features and target values horizontally (column-wise)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Save the fitted preprocessing object to a file for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

