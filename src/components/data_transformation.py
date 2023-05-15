import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,select_lasso

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self,features):
        try:

            num_pipeline= Pipeline(
                steps=[
                ("scaler",StandardScaler())
                ]
            )
            
            logging.info("In data transformer object")

            preprocessing = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,features)
                ]
            )
            
            return preprocessing

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,X_train_set,X_test_set,y_train_set,y_test_set):

        try:
            X_train=pd.read_csv(X_train_set)
            X_test=pd.read_csv(X_test_set)
            y_train=pd.read_csv(y_train_set)
            y_test=pd.read_csv(y_test_set)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            #preprocessing_obj=self.get_data_transformer_object()

            target_column_name="status"
            numerical_columns = select_lasso(X_train,y_train)
            
            

            preprocessing_obj=self.get_data_transformer_object(numerical_columns)

            X_train = X_train[numerical_columns]
            y_train = y_train[target_column_name].values

            X_test = X_test[numerical_columns]
            y_test = y_test[target_column_name].values

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            X_train=preprocessing_obj.fit_transform(X_train)
            X_test=preprocessing_obj.transform(X_test)


            '''train_arr = np.c_[
                input_feature_train_arr, np.array(y_train)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(y_test)]'''

            

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                X_train,
                X_test,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        