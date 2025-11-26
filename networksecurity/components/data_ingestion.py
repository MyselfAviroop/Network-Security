from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pymongo
from dotenv import load_dotenv

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """Read data from MongoDB collection and export as DataFrame"""
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            logging.info(f"Fetched {df.shape[0]} rows and {df.shape[1]} columns from MongoDB.")
            df = pd.DataFrame(list(collection.find()))
            print("DEBUG: MongoDB returned rows =", df.shape[0])

            df.drop(columns=["_id"], inplace=True)

            df.replace(to_replace="na", value=np.nan, inplace=True)
            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, df: pd.DataFrame) -> str:
        """Export DataFrame into the feature store (raw data CSV)"""
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            df.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Exported data to feature store at {feature_store_file_path}")
            return feature_store_file_path

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, df: pd.DataFrame) -> tuple:
        """Split the dataset into train and test sets"""
        try:
            train_set, test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            logging.info("Successfully split the data into train and test sets.")

            # Ensure directories exist
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.test_file_path), exist_ok=True)

            # Save CSVs
            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)

            logging.info(f"Train data saved at {self.data_ingestion_config.train_file_path}")
            logging.info(f"Test data saved at {self.data_ingestion_config.test_file_path}")

            return (
                self.data_ingestion_config.train_file_path,
                self.data_ingestion_config.test_file_path
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Orchestrates the entire data ingestion workflow"""
        try:
            # Step 1: Load data from MongoDB
            df = self.export_collection_as_dataframe()

            # Step 2: Save DataFrame to feature store
            self.export_data_into_feature_store(df)

            # Step 3: Split into train/test
            train_path, test_path = self.split_data_as_train_test(df)

            # Step 4: Create artifact object
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=train_path,
                test_file_path=test_path
            )

            logging.info(f"Data Ingestion completed successfully: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
