import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    X_train_data_path: str=os.path.join('artifacts',"xtrain.csv")
    X_test_data_path: str=os.path.join('artifacts',"xtest.csv")
    y_train_data_path: str=os.path.join('artifacts',"ytrain.csv")
    y_test_data_path: str=os.path.join('artifacts',"ytest.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/parkinsons.csv')
            logging.info('Read the dataset as dataframe')
            X = df.drop(["name","status"],axis=1) # name not needed
            y = df["status"]  

            os.makedirs(os.path.dirname(self.ingestion_config.X_train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

            X_train.to_csv(self.ingestion_config.X_train_data_path,index=False,header=True)
            X_test.to_csv(self.ingestion_config.X_test_data_path,index=False,header=True)
            y_train.to_csv(self.ingestion_config.y_train_data_path,index=False,header=True)
            y_test.to_csv(self.ingestion_config.y_test_data_path,index=False,header=True)

            #test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.X_train_data_path,
                self.ingestion_config.X_test_data_path,
                self.ingestion_config.y_train_data_path,
                self.ingestion_config.y_test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    X_train,X_test,y_train,y_test=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    X_train, X_test, y_train, y_test,_ = data_transformation.initiate_data_transformation(X_train,X_test,y_train,y_test)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(X_train,X_test,y_train,y_test))