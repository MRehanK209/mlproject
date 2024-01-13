import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # For Creating a Trasformation pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_obj(self):
        """
            This function is responsible for Data Transformation 
        """
        try :
            numerical_cols = ['writing_score', "reading_score"]
            categorical_cols = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'  
            ]

            num_pipline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical Pipeline Initiated")

            cat_pipelines = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot", OneHotEncoder()),
                    ("scalar", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical pipeline Initiated")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipline, numerical_cols),
                    ("categorical_pipline", cat_pipelines, categorical_cols)
                ]
            )
            return preprocessor
        except Exception as e :
            CustomException(e,sys)
    
    def initiate_data_tranformation(self,train_data_path,test_data_path):

        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Read Train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_obj()
            target_column_name = "math_score"

            #numerical_cols = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # print("test_df.shape " , test_df.shape)
            # print("train_df.shape " , train_df.shape)    

            # print("input_feature_test_df.shape " , input_feature_test_df.shape)
            # print("input_feature_train_df.shape " , input_feature_train_df.shape)
            logging.info(
                f"applying preprocessing object on Training and Test Dataframes"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # print("length of input feature training array", len(input_feature_train_arr))
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(
                "Saved Preprocessing Object"
            )

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            