import os
import sys
from dataclasses import dataclass

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models,accuracy_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,X_train,X_test,y_train,y_test):
        try:
            logging.info("Split training and test input data")
            
            models = {
                "Random Forest": RandomForestClassifier(criterion="entropy"),
                "XGBRegressor": XGBClassifier(),
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = accuracy_score(y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)