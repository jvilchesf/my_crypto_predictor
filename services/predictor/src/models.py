import pandas as pd
import os
from lazypredict.Supervised import LazyRegressor
from loguru import logger
from typing import Optional
import optuna

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import HuberRegressor

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

class HuberRegressorWraperWithHyperparameterTuning:

    def __init__(
            self,
            hyperparam_search_trials: Optional[int] = 0,
            hyperparam_splits: Optional[int] = 3
        ):

        self.model: HuberRegressor = HuberRegressor()
        self._do_hyperparam_tuning = hyperparam_search_trials > 0
        self.hyperparam_splits = hyperparam_splits


    def fit( 
            self,
            X,
            y
        ):
        
        if self._do_hyperparam_tuning == 0:
            logger.info(f"Fiting the model without hyperparameter tunning ")
            self.mode.fit(X, y)
        else:
            #Implement hyper parameter tuning
            logger.info(f"Finding Hyperparameters for the model with optuna library")

    def _find_best_hyperparams(
            self,
            X_train: pd.Dataframe,
            y_train: pd.Series
    ) -> dict:
        
        """"
        Find the best hyperparamteters for the model using Bayesian optimization
        """

        def objective(trial: optuna.Trial) -> float:

            """"
            
            """             
            # We ask Optuna to same the next set of hyperparameters for huber regressor
            # these are our candidates for this trial

            params = {
                'epsilon' : trial.suggest_uniform('epsilon', 0.1, 1),
                'max_iter' : trial.suggest_int('max_iter', 100, 1000),
                'alpha' : trial.suggest_uniform('alpha', 0.01, 0.1),
                'warm_start' : trial.suggest_categorical('warm_start', [True, False]),
                'fit_intercept' : trial.suggest_categorical('fit_intercept', [True, False]),
                'tol' : trial.suggest_uniform('tol', 1e-4, 1e-2)
            }

            # We fit the model with the given hyperparameters
            model = HuberRegressor(**params)
            model.fit(X_train, y_train)

            # We return the negative mean absolute error as optuna seeks to minimize the objective
            


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