import pandas as pd
import numpy as np
import os
from lazypredict.Supervised import LazyRegressor
from loguru import logger
from typing import Optional
import optuna

import mlflow

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn import linear_model

from sklearn.compose import TransformedTargetRegressor
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
        return X["close"]


class ModelWithHyperparameterTuning:
    """ "
    Fit the model with Hyperparameters tunnings
    """

    def __init__(
        self,
        model_name: str,
        hyperparam_search_trials: Optional[int] = 0,
        hyperparam_splits: Optional[int] = 3,
    ):
        """ "
        Initialize the model

        Args:
            model_name: Receive the name of the model that needs to be created
            hyperparam_search_trials: Optional integer paramenter defined in settings.env define how many iterations will be execute by optuna to find best hyperparameters
            hyperparam_splits: Optional integer paramenter defined in settings.env that tells how many times the train dataset will be splitted in train and validation datasets
        """

        self.model_name = model_name
        self.pipeline = self._get_pipeline()

        self.hyperparam_search_trials = hyperparam_search_trials
        self._do_hyperparam_tuning = (
            hyperparam_search_trials > 0
        )  # if hyperparam_search_trials > 0 means there are intentions of hyperparameter tunning, it is used after in the 'fit' function
        self.hyperparam_splits = hyperparam_splits

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.pipe.model.predict(X)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """ "
        Fit function of the model, it will look for hyperparameters depending on the parametric _do_hyperparam_tuning variable.

        Args:
            X: train dataset
            y: target train dataset
        """

        if self._do_hyperparam_tuning == 0:
            logger.info("Fiting the model without hyperparameter tunning ")
            return self.pipeline.fit(X, y)
        else:
            # Implement hyper parameter tuning
            logger.info(
                f"Finding Hyperparameters for the model with optuna library, {self.hyperparam_search_trials}"
            )
            model_hyperparams = self._find_best_hyperparams(X, y)
            logger.info(
                f"Hyperparameter tunning with optuna is ready!, best trial: {model_hyperparams}"
            )
            self.pipeline = self._get_pipeline(model_hyperparams)
            logger.info(
                f"Start fitting the {self.model_name} model with the best hyperparameters"
            )
            return self.pipeline.fit(X, y)

    def _get_pipeline(
        self,
        model_hyperparams: Optional[dict] = None,
    ) -> Pipeline:
        """ "
        Function to return dinamically pipelines depending on the model called and the hyperparameters

        Args:
            model_hyperparams: Optional dictionary with the model hyperparameters
        """

        # Create different pipelines depending on the model called

        if model_hyperparams is None:
            if self.model_name == "HuberRegressor":
                model = HuberRegressor()
            elif self.model_name == "TransformedTargetRegressor":
                model = TransformedTargetRegressor(
                    regressor=LinearRegression(), func=np.log, inverse_func=np.exp
                )
            elif self.model_name == "LinearRegression":
                model = LinearRegression()
            elif self.model_name == "LassoLarsIC":
                model = linear_model.LassoLarsIC()
        else:
            if self.model_name == "HuberRegressor":
                model = HuberRegressor(**model_hyperparams)
            elif self.model_name == "TransformedTargetRegressor":
                model = TransformedTargetRegressor(**model_hyperparams)
            elif self.model_name == "LinearRegression":
                model = LinearRegression(**model_hyperparams)
            elif self.model_name == "LassoLarsIC":
                model = linear_model.LassoLarsIC(**model_hyperparams)

        return Pipeline(steps=[("scaler", StandardScaler()), ("model", model)])

    def _find_best_hyperparams(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """ "
        Find the best hyperparamteters for the model using Bayesian optimization
        """

        def objective(trial: optuna.Trial) -> float:
            """ "
            Core of the model, here is where we find the best hyperparameters values. and is where using 'TimeSerieSplit' with divide the train dataset into
            validation in train folds to look up for the best hyperparameters values.
            """
            # We ask Optuna to same the next set of hyperparameters for huber regressor
            # these are our candidates for this trial

            params = get_model_params_dict(self.model_name, trial)

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

                # Update the pipeline defined in the class constructor with the new params. in that way we can fit the mode and find new parameters with an updated model
                self.pipeline = self._get_pipeline(params)

                # Fit model
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
        study_name = f"{self.model_name}_study"
        study = optuna.create_study(study_name=study_name)
        study.optimize(objective, n_trials=self.hyperparam_search_trials, timeout=600)
        return study.best_trial.params


def fit_lazy_regresor_n_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    mlflow_tracking_uri: str
) -> pd.DataFrame:
    """
    This function evaluate the input train dataset over a set of model

    Args:
        X_train: DataFrame

    Return:
        df_lazy_predictor: DataFrame with all possible models
    """

    # Unset enviromental variable for mlflow for the process to think I'm not using mlflow
    # I've skipped to run one experiment for each model in mlflow
    del os.environ["MLFLOW_TRACKING_URI"]

    logger.info("Using lazy predict to get the model fit better to the data")

    logger.info(f"{mlflow_tracking_uri}")

    reg = LazyRegressor(verbose=False, ignore_warnings=True, custom_metric= mean_absolute_error)
    
    logger.info(f"reg: {reg}")
    models, _ = reg.fit(X_train, X_test, y_train, y_test)

    logger.info(f"models: {models}")
    #Set back enviromental variable for mlflow
    os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri

    return models


def get_model_names(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    top_n_models: int,
    mlflow_tracking_uri: str
) -> list[str]:
    """ "
    This function will return a list with top N models

    Args:
        X_train: Pandas Dataframe object with all training data used to run lazy model function
        y_train: Pandas Dataframe object with all target train data
        X_test: Pandas Dataframe object with all testing data used to run lazy model function
        y_test: Pandas Dataframe object with all target train data
        top_n_models: Parameter defined in settings.env will determine how many models will be compared
    """

    # Train a set of models and see which one perform the best
    # Use lazy predict to evaluate the dataset with a list of models
    df_lazy_predictor = fit_lazy_regresor_n_models(X_train, y_train, X_test, y_test, mlflow_tracking_uri)

    # Reset index to save all columns in mlflow .json
    df_lazy_predictor = df_lazy_predictor.reset_index()

    # Log lazy predictor table in mlflow
    logger.info("Saving lazy predictor models performance into mlflow")
    mlflow.log_table(
        df_lazy_predictor, artifact_file="models_evaluation_lazy_predictor.json"
    )

    return df_lazy_predictor[:top_n_models]


def model_execution(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test_compare: pd.DataFrame,
    y_test_compare: pd.DataFrame,
    model_name: str,
    hyperparam_search_trials: int,
    hyperparam_splits: int,
) -> int:
    """ "
    This function will create and fit the model

    Args:
        model_name: Contains an string value with the model name needs to be ran.
        hyperparam_search_trials: Variable that defines how many trials will be executed by optuna.
        hyperparam_splits: It defines how many times the data will be splited on each

    Return:
        It will return the MAE (mean absolute error)
    """
    # Create a custom model
    model = ModelWithHyperparameterTuning(
        model_name, hyperparam_search_trials, hyperparam_splits
    )
    logger.info(f"Comparing models: Model {model_name} created")
    # Fit model
    logger.info(f"Comparing models: Model {model_name} Fitting in progress")
    model = model.fit(X_train, y_train)
    logger.info(f"Comparing models: Model {model_name} Fitting Ready")
    # evaluate model with test data
    y_predict = model.predict(X_test_compare)
    logger.info(f"Comparing models: Model {model_name} Predicting in progress")
    # calculate mae
    mae = mean_absolute_error(y_test_compare, y_predict)
    logger.info(f"Comparing models: Model {model_name} MAE: {mae}")
    return mae


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test_compare: pd.DataFrame,
    y_test_compare: pd.DataFrame,
    models_name: pd.Series | str,
    hyperparam_search_trials: int,
    hyperparam_splits: int,
) -> ModelWithHyperparameterTuning | str:
    """ "
    This function receive a list of models to compare, it will run one by one and save their mae to finally compare and
    select the model with best performance.

    Model can receive a list of models or a predefined model

    Args:
        model_name: List with top n models | string with model name already set.
    """

    # If model_name is a lsit of n top model, go through the list and calculate mae for each of the three top models
    # Hyperparameter tunning will be nedded to compare
    mae_list = {}
    count = 0

    if isinstance(models_name, pd.Series):
        for model_name in models_name:
            # Check if model is in the list of models we have for training
            if model_name in [
                "HuberRegressor",
                "TransformedTargetRegressor",
                "LinearRegression",
                "LassoLarsIC",
            ]:
                # Save model name in mlflow
                mlflow.log_param(f"compared_model_name_{count + 1}", model_name)

                # Select model
                mae = model_execution(
                    X_train,
                    y_train,
                    X_test_compare,
                    y_test_compare,
                    model_name,
                    hyperparam_search_trials,
                    hyperparam_splits,
                )

                # Save model name and mae in a dictionary
                mae_list[model_name] = mae
                count += 1
            else:
                logger.info(
                    f"Compare models: Model {model_name} not found in the models list"
                )
                if len(models_name) == count: 
                    logger.info("First top n models are not included as a trainable model, we will use HuberRegressor as fallback")
                    mae_list = {'HuberRegressor': 0}
                    break
                continue
        # Get the model name with the lowest mae
        best_model = min(mae_list, key=mae_list.get)

        # Save best model name in mlflow
        mlflow.log_param("best_model_name", best_model)
        # Save best model mae in mlflow
        mlflow.log_param("best_model_mae", mae_list[best_model])
        # Save all models mae in mlflow
        mlflow.log_param("all_models_mae", mae_list)

        logger.info(f"Compare models: Best model is {best_model}")
        # Return an instance of the best model
        return ModelWithHyperparameterTuning(
            best_model, hyperparam_search_trials, hyperparam_splits
        )

    elif models_name == "HuberRegressor":
        logger.info(
            "Compare models: Model selected preselected manually: HuberRegressor"
        )
        return ModelWithHyperparameterTuning(
            models_name, hyperparam_search_trials, hyperparam_splits
        )
    else:
        # If we get here, we didn't find HuberRegressor
        logger.warning(
            "Model not found in the models list. Using HuberRegressor as fallback."
        )
        return ModelWithHyperparameterTuning(
            "HuberRegressor", hyperparam_search_trials, hyperparam_splits
        )


def get_model_params_dict(
    model_name: str,
    trial: optuna.Trial,
) -> dict:
    """ "
    This function will return a dictionary with the model parameters

    Args:
        model_name: str with the model name
        trial: optuna.Trial object
    """

    # Define main dictionary with all model parameters

    if model_name == "TransformedTargetRegressor":
        return {
            "regressor": LinearRegression(),  # Always use LinearRegression as the base regressor
            "transformer": None,
            "func": trial.suggest_categorical(
                "func", [np.log, np.sqrt, np.arcsinh]
            ),  # Use log transformation
            "inverse_func": trial.suggest_categorical(
                "inverse_func", [np.exp, np.sqrt, np.arcsinh]
            ),  # Use exp as inverse
            "check_inverse": trial.suggest_categorical("check_inverse", [True, False]),
        }

    if model_name == "LinearRegression":
        return {
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "positive": trial.suggest_categorical("positive", [True, False]),
            "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
        }

    if model_name == "LassoLarsIC":
        return {
            "criterion": trial.suggest_categorical("criterion", ["aic", "bic"]),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "verbose": trial.suggest_categorical("verbose", [False, True]),
            "precompute": trial.suggest_categorical(
                "precompute", ["auto", True, False]
            ),
            "max_iter": trial.suggest_int("max_iter", 500, 1000),
            "eps": trial.suggest_uniform("eps", 1e-16, 1e-15),
            "copy_X": trial.suggest_categorical("copy_X", [True, False]),
            "positive": trial.suggest_categorical("positive", [False, True]),
            "noise_variance": trial.suggest_categorical("noise_variance", [None]),
        }
    if model_name == "HuberRegressor":
        return {
            "epsilon": trial.suggest_uniform(
                "epsilon", 1.00, 99999999
            ),  # More focused range
            "alpha": trial.suggest_uniform("alpha", 0.01, 1.0),  # Smaller alpha range
            "max_iter": trial.suggest_int(
                "max_iter", 100, 1000
            ),  # Reduced max iterations
            "tol": trial.suggest_uniform("tol", 1e-4, 1e-2),  # More precise tolerance
            "warm_start": trial.suggest_categorical("warm_start", [True, False]),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
