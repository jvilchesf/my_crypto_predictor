import pandas as pd
import os
from lazypredict.Supervised import LazyRegressor
from loguru import logger
from typing import Optional
import optuna

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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

    """"
    Fit the model with Hyperparameters tunnings
    """

    def __init__(
            self,
            hyperparam_search_trials: Optional[int] = 0,
            hyperparam_splits: Optional[int] = 3
        ):
        """"
        Initialize the model

        Args:
            hyperparam_search_trials: Optional integer paramenter defined in settings.env define how many iterations will be execute by optuna to find best hyperparameters
            hyperparam_splits: Optional integer paramenter defined in settings.env that tells how many times the train dataset will be splitted in train and validation datasets
        """

        self.pipeline = self._get_pipeline()

        self.hyperparam_search_trials = hyperparam_search_trials
        self._do_hyperparam_tuning = hyperparam_search_trials > 0 # if hyperparam_search_trials > 0 means there are intentions of hyperparameter tunning, it is used after in the 'fit' function
        self.hyperparam_splits = hyperparam_splits

    def predict(
            self,
            X: pd.DataFrame
    ) -> pd.Series:
        
        return self.pipe.model.predict(X)

    def fit( 
            self,
            X,
            y
        ):
        """"
        Fit function of the model, it will look for hyperparameters depending on the parametric _do_hyperparam_tuning variable.

        Args:
            X: train dataset
            y: target train dataset
        """
        
        if self._do_hyperparam_tuning == 0:
            logger.info(f"Fiting the model without hyperparameter tunning ")
            return self.pipeline.fit(X, y)
        else:
            #Implement hyper parameter tuning
            logger.info(f"Finding Hyperparameters for the model with optuna library, {self.hyperparam_search_trials}")
            model_hyperparams = self._find_best_hyperparams(X, y)
            logger.info(f"Hyperparameter tunning with optuna is ready!, best trial: {model_hyperparams}")
            self.pipeline = self._get_pipeline(model_hyperparams)
            logger.info(f"Start fitting the selected model with the best hyperparameters")
            return self.pipeline.fit(X, y)
        
    def _get_pipeline(
            self,
            model_hyperparams: Optional[dict] = None
        ) -> Pipeline:
        
        """"
        Standar function to return pipeline definition depending if hyperparam_search_trials is defined or not
        """

        if model_hyperparams is None:
            return Pipeline(
                    steps = [
                        ('scaler', StandardScaler()),
                        ('model', HuberRegressor())
                ]
            )
        else: 
            return Pipeline(
                    steps = [
                        ('scaler', StandardScaler()),
                        ('model', HuberRegressor(**model_hyperparams))
                    ])

    def _find_best_hyperparams(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series
    ) -> dict:
        
        """"
        Find the best hyperparamteters for the model using Bayesian optimization
        """

        def objective(trial: optuna.Trial) -> float:

            """"
            Core of the model, here is where we find the best hyperparameters values. and is where using 'TimeSerieSplit' with divide the train dataset into 
            validation in train folds to look up for the best hyperparameters values.
            """             
            # We ask Optuna to same the next set of hyperparameters for huber regressor
            # these are our candidates for this trial

            params = {
                'epsilon' : trial.suggest_uniform('epsilon', 1.00, 99999999),  # More focused range
                'alpha' : trial.suggest_uniform('alpha', 0.01, 1.0),  # Smaller alpha range
                'max_iter' : trial.suggest_int('max_iter', 100, 1000),    # Reduced max iterations
                'tol' : trial.suggest_uniform('tol', 1e-4, 1e-2),        # More precise tolerance
                'warm_start' : trial.suggest_categorical('warm_start', [True, False]),
                'fit_intercept' : trial.suggest_categorical('fit_intercept', [True, False]),
            }

            # Split the dataset into n_splits folds 
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=self.hyperparam_splits)

            mae_score = []

            for X_train_indx, X_val_indx in tscv.split(X_train):

                    # Split training dataset in: train and validation dataset
                    X_train_fold = X_train.iloc[X_train_indx]
                    X_val_fold = X_train.iloc[X_val_indx]

                    y_train_fold = y_train.iloc[X_train_indx]
                    y_val_fold = y_train.iloc[X_val_indx]

                    #Update the pipeline defined in the class constructor with the new params. in that way we can fit the mode and find new parameters with an updated model
                    self.pipeline = self._get_pipeline(params)

                    #Fit model
                    self.pipeline.fit(X_train_fold, y_train_fold)

                    # Make predictions
                    y_pred = self.pipeline.predict(X_val_fold)

                    # Calculate error
                    mae = mean_absolute_error(y_val_fold, y_pred)
                    
                    # Save error in a list to average and return
                    mae_score.append(mae)
            
            import numpy as np

            return np.mean(mae_score)

        # Create an optuna study, it is a concept of the optuna library. Each study has several trial iterations to find the best hyperparameters
        study = optuna.create_study()
        study.optimize(objective, n_trials=self.hyperparam_search_trials, timeout=600)
        return study.best_trial.params


def fit_lazy_regresor_n_models(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame
    ) -> pd.DataFrame :
    """
    This function evaluate the input train dataset over a set of model

    Args:
        X_train: DataFrame

    Return: 
        df_lazy_predictor: DataFrame with all possible models
    """

    #Unset enviromental variable for mlflow for the process to think I'm not using mlflow
    #I've skipped to run one experiment for each model in mlflow
    del os.environ['MLFLOW_TRACKING_URI'] 
    
    logger.info(f"Using lazy predict to get the model fit better to the data")

    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric= mean_absolute_error)
    models, _ = reg.fit(X_train, X_test, y_train, y_test)

    #Set back enviromental variable for mlflow
    os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:8889'

    return models

def get_model(
        df_lazy_predictor: pd.DataFrame
    ) -> HuberRegressorWraperWithHyperparameterTuning | str: 

    """"
    Receive all model candidates output from lazypredictor in descending order, from best to worst

    Args:
        df_lazy_predictor: Dataframe with all candidates
    """
    for index, row in df_lazy_predictor.iterrows():
        if row['Model'] == 'HuberRegressor':
            logger.info(f"Model selected to be trained: {row['Model']}")
            return HuberRegressorWraperWithHyperparameterTuning
        else:
            logger.info(f"Hey we've not implemented {row['Model']}. we will go for the next option")
            logger.info(f"Position= {index+1}, Name= {row['Model']}, MAE= {row['mean_absolute_error']}")
            continue
    
    # If we get here, we didn't find HuberRegressor
    logger.warning("HuberRegressor not found in the models list. Using LinearRegression as fallback.")
    return "Error, no model was founded"  # Fallback to HuberRegressor anyway