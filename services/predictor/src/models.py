import pandas as pd
import os
from lazypredict.Supervised import LazyRegressor
from loguru import logger

from sklearn.metrics import mean_absolute_error

class BaselineModel:

    def __init__(self):
        """
        Initialize the model parameters
        """
        pass

    def fit(self, X, Y):
        """
        Fit the model to the training data
        """
        pass

    def predict(self, X) -> pd.Series:
        """
        Predict the target for the given data
        """
        return X['close']
    
def fit_lazy_regresor_n_models(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame
    ) -> pd.DataFrame :
    """
    This function evaluate the input train dataset over a set of model

    Args:
        X_train: Dataframe

    Return: 
        df_lazy_predictor: Dataframe with all possible models
    """

    #Unset enviromental variable for mlflow for the process to think I'm not using mlflow
    #I've skipped to run one experiment for each model in mlflow
    del os.environ['MLFLOW_TRACKING_URI'] 
    
    logger.info(f"Using lazy predict to get the model fit better to my data")

    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric= mean_absolute_error)
    models, _ = reg.fit(X_train, X_test, y_train, y_test)

    #Set back enviromental variable for mlflow
    os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:8889'

    return models