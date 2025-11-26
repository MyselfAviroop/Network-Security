import sys
import numpy as np
import pandas as pd
import os
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.constant.training_pipeline import (
    TARGET_COLUMN,
    DATA_TRANSFORMATION_IMPUTER_PARAMS
)

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from networksecurity.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object
)


class DataTransformation:

    def __init__(self,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info("Creating KNN Imputer...")
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            pipeline = Pipeline(steps=[("imputer", imputer)])
            logging.info("KNN Imputer pipeline created.")
            return pipeline
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation...")

            # Read validated datasets
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Split input/target features
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            # Preprocessor
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            transformed_input_train = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test = preprocessor_object.transform(input_feature_test_df)

            # Combine features + target
            train_arr = np.c_[
                transformed_input_train,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                transformed_input_test,
                np.array(target_feature_test_df)
            ]

            # Save transformed arrays
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            # Save preprocessing object
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_object
            )

            # Return artifact
            transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )

            logging.info(f"Data Transformation Artifact: {transformation_artifact}")

            return transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
