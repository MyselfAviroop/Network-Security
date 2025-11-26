import yaml
from networksecurity.logging.logger import logging
from networksecurity.exception import exception
import os, sys
import numpy as np
import pickle
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def read_yaml_file(file_path: str) -> dict:
    """
    Read YAML file and return its content as a dictionary.
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise exception.NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, data: dict = None):
    """
    Write a dictionary to a YAML file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            if data is not None:
                yaml.safe_dump(data, yaml_file)
    except Exception as e:
        raise exception.NetworkSecurityException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save a numpy array to a .npy file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise exception.NetworkSecurityException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load a numpy array from a file.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise exception.NetworkSecurityException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object using pickle.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise exception.NetworkSecurityException(e, sys)


def load_object(file_path: str) -> object:
    """
    Load a Python object saved with pickle.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exist")
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise exception.NetworkSecurityException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models: dict, params: dict):
    """
    Evaluate multiple models using GridSearchCV and return performance report.
    """
    try:
        report = {}

        for model_name in models.keys():
            model = models[model_name]
            param_grid = params[model_name]

            gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
            gs.fit(x_train, y_train)

            best_model = gs.best_estimator_
            best_model.fit(x_train, y_train)

            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = {
                "train_score": train_score,
                "test_score": test_score
            }

        return report

    except Exception as e:
        raise exception.NetworkSecurityException(e, sys)
