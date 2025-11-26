import os
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from networksecurity.utils.main_utils.utils import load_object

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from networksecurity.utils.main_utils.utils import (
    load_numpy_array_data,
    save_object,
    evaluate_models
)

from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel


class ModelTrainer:

    def __init__(self,
                 model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            logging.info("Model Trainer initialized.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, x_train, y_train, x_test, y_test):

        try:
            models = {
                "RandomForest": RandomForestClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "DecisionTree": DecisionTreeClassifier(),
                "LogisticRegression": LogisticRegression(max_iter=2000),
                "AdaBoost": AdaBoostClassifier()
            }

            params = {
                "RandomForest": {"n_estimators": [64, 128, 256]},
                "DecisionTree": {
                    "criterion": ['gini', 'entropy'],
                    "splitter": ['best', 'random']
                },
                "AdaBoost": {"n_estimators": [32, 64, 128]},
                "GradientBoosting": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [64, 128]
                },
                "LogisticRegression": {"C": [1, 10]}
            }

            # Evaluate models
            model_report = evaluate_models(
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test,
                models=models, params=params
            )

            # Select best model using test_score
            best_model_name = max(model_report, key=lambda m: model_report[m]["test_score"])
            best_model = models[best_model_name]

            logging.info(f"Best model selected: {best_model_name}")

            # Re-train best model fully
            best_model.fit(x_train, y_train)

            # Metrics
            y_train_pred = best_model.predict(x_train)
            train_metrics = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            y_test_pred = best_model.predict(x_test)
            test_metrics = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # Save network model wrapper
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir, exist_ok=True)

            network_model = NetworkModel(preprocesssor=preprocessor, model=best_model)


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=network_model
            )

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self):

        try:
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(x_train, y_train, x_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
