import logging
from abc import ABC, abstractmethod

from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

class Evalauation(ABC):
    
    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray, y_pred:np.ndarray):

        pass

class MSE(Evalauation):

    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true,y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e
        

class R2(Evalauation):

    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_true,y_pred)
            logging.info("MSE: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2Score: {}".format(e))
            raise e
        
class RMSE(Evalauation):

    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true,y_pred)
            logging.info("MSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e



