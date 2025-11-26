from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
import sys
import os
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkModel:
    def __init__(self, preprocesssor,model):
      try:
            self.preprocesssor=preprocesssor
            self.model=model
      except Exception as e:
            raise NetworkSecurityException(e,sys)
      
    def predict(self,x):
        try:
            x_transform=self.preprocesssor.transform(x)
            y_pred=self.model.predict(x_transform)
            return y_pred
        except Exception as e:
            raise NetworkSecurityException(e,sys)