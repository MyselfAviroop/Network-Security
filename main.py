from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from networksecurity.exception.exception import NetworkSecurityException
import sys


if __name__ == "__main__":
    try:
        # ---------------- TRAINING PIPELINE CONFIG ----------------
        pipeline_config = TrainingPipelineConfig()

        # ---------------- DATA INGESTION ----------------
        ingestion_config = DataIngestionConfig(pipeline_config)
        ingestion = DataIngestion(ingestion_config)
        ingestion_artifact = ingestion.initiate_data_ingestion()
        print("\n✔ Data Ingestion Completed")

        # ---------------- DATA VALIDATION ----------------
        validation_config = DataValidationConfig(pipeline_config)
        validation = DataValidation(
            data_ingestion_artifact=ingestion_artifact,
            data_validation_config=validation_config
        )
        validation_artifact = validation.initiate_data_validation()
        print("\n✔ Data Validation Completed")

        # ---------------- DATA TRANSFORMATION ----------------
        transformation_config = DataTransformationConfig(pipeline_config)
        transformation = DataTransformation(
        data_validation_artifact=validation_artifact,
        data_transformation_config=transformation_config
    )

        transformation_artifact = transformation.initiate_data_transformation()
        print("\n✔ Data Transformation Completed")

        # ---------------- MODEL TRAINER ----------------
        trainer_config = ModelTrainerConfig(pipeline_config)
        trainer = ModelTrainer(
            model_trainer_config=trainer_config,
            data_transformation_artifact=transformation_artifact
        )
        trainer_artifact = trainer.initiate_model_trainer()
        print("\n✔ Model Training Completed")

        print("\n🎉 TRAINING PIPELINE FINISHED SUCCESSFULLY 🎉")

    except Exception as e:
        raise NetworkSecurityException(e, sys)
