import pandas as pd
from loguru import logger
from typing import Optional

from risingwave import RisingWave, RisingWaveConnOptions, OutputFormat
from ydata_profiling import ProfileReport
import mlflow
from mlflow.data.pandas_dataset import PandasDataset

from config import Settings
from models import BaselineModel, get_model_names, compare_models
from names import get_experiment_name
from transform_clean_data import SplitData, TransformCleanData

from sklearn.metrics import mean_absolute_error

def generate_exploratory_data_analysis(
    ts_data: pd.DataFrame,
    output_file_path: str
    ):
    """
    Generate exploratory data analysis for the given time-series data, it saves an html report in the given path

    Args:
        ts_data: pd.DataFrame,
        output_file_path: str = path where the html report will be saved
    """
    profile = ProfileReport(ts_data, tsmode=True, sortby="window_start_ms", title="Technical Indicators Prediction EDA")
    profile.to_file(output_file=output_file_path)
    
def load_data_from_risingwave(
    rw_host: str,
    rw_port: int,
    rw_user: str,
    rw_database: str,
    symbol: str,
    days_in_past: int,
    candle_seconds: int
    ) -> pd.DataFrame:
    """
    Load technical indicators data from risingwave for the given symbol and days in the past

    Args:
        rw_host: str,
        rw_port: int,
        rw_user: str,
        rw_database: str,
        symbol: str,
        days_in_past: int,
        candle_seconds: int

    Returns:
        pd.DataFrame: A pandas dataframe containing the technical indicators data
    """
        
    # Connect to RisingWave instance on localhost with named parameters
    rw = RisingWave(
        RisingWaveConnOptions.from_connection_info(
            host=rw_host, port=rw_port, user=rw_user, password="", database=rw_database
        )
    )

    query = f"""
        SELECT
         * 
        FROM public.technical_indicators
        WHERE symbol = '{symbol}' 
        AND to_timestamp(window_start_ms / 1000) >= now() - interval '{days_in_past} days'
        AND candle_seconds = {candle_seconds}
        ORDER BY window_start_ms  
        """

    # Fetch data from RisingWave
    result: pd.DataFrame = rw.fetch(query, 
        format=OutputFormat.DATAFRAME)
    
    return result

def train(
    rw_host: str,
    rw_port: int,
    rw_user: str,
    rw_database: str,
    symbol: str,    
    days_in_past: int,
    candle_seconds: int,
    prediction_horizon_seconds: int,
    n_rows_for_data_profiling: int,
    mlflow_tracking_uri: str,
    train_test_split_ratio: float, 
    list_features: list[str],
    hyperparam_search_trials: int,
    hyperparam_splits: int,
    model_name: Optional[str],
    top_n_models: Optional[int],
    treshold_null_values: float,
    treshold_select_model: float
    ):

    """
    train a predictor model forh the give data pair and data, and if the model is good enough it will be saved to the model registry
    """

    logger.info(f"Starting training for {symbol} with {days_in_past} days in the past and {candle_seconds} seconds per candle")

    # 0. Set MLflow 
    # - tracking uri
    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

    experiment_name = get_experiment_name(symbol, days_in_past, candle_seconds)
    # Create a new MLflow Experiment
    mlflow.set_experiment(experiment_name)

    #Things we have to log with MLflow 
    # - EDA report
    # - Sample of the data
    # - Model parameters
    # - Model performance
    # - Model artifacts (hotml)

    with mlflow.start_run(nested = True):

        logger.info(f"Start loggin data to MLflow")

        # 1. Fetch data from RisingWave for the given symbol and days in the past
        logger.info(f"Loading data from RisingWave for the given symbol and days in the past")
        ts_data = load_data_from_risingwave(rw_host, rw_port, rw_user, rw_database, symbol, days_in_past, candle_seconds)
        logger.info(f"Loaded {len(ts_data)} rows of data from RisingWave")
        
        # 2. Add target column
        logger.info(f"Adding target column")
        ts_data['target'] = ts_data['close'].shift(-prediction_horizon_seconds//candle_seconds)

        #Filter just necessary features
        ts_data = ts_data[list_features]

        # Log data to MLflow
        # Wrap dataframe as dataset
        dataset: PandasDataset = mlflow.data.from_pandas(ts_data)
        mlflow.log_input(dataset, context="training")

        # Log dataset size
        mlflow.log_param("ts_data_shape", ts_data.shape)

        # 3. Validate the data  with greate expectations
        transform_clean_data = TransformCleanData(ts_data)
        logger.info("Validating data with GEX (great expectations) python library")
        ts_data = transform_clean_data.validate_data(treshold_null_values)

        # 4. Profile the data using ydata-profiling
        ts_data_for_profiling = ts_data.head(n_rows_for_data_profiling) if n_rows_for_data_profiling else ts_data
        logger.info("Profiling the data using ydata-profiling")
        generate_exploratory_data_analysis(ts_data_for_profiling, output_file_path="./prediction_crypto_eda.html")
        # Log html to MLflow
        mlflow.log_artifact(local_path= "./prediction_crypto_eda.html", artifact_path="eda_report")

        # 5. Split it into train and test
        logger.info(f"Splitting data into train and test with ratio {train_test_split_ratio}")
        
        split_data = SplitData(ts_data)
        X_train, y_train, X_test, y_test = split_data.split_train_test_datasets(train_test_split_ratio)
        X_test_compare, y_test_compare, X_test, y_test = split_data.split_test_compare_datasets(X_test, y_test)

        # 6. Baseline model
        logger.info("Building a dummy baseline model")
        baseline_model = BaselineModel()
        y_pred_baseline_model = baseline_model.predict(X_test)

        # 7. Log baseline performance
        # Define an error metric MEA (mean absolute error)
        mae_baseline_model = mean_absolute_error(y_test, y_pred_baseline_model)
        mlflow.log_metric("mae_baseline_model", mae_baseline_model)
        logger.info(f"MAE of the baseline model is {mae_baseline_model}")

        # 8. Get best model using lazy predictor, a function will be run to receive a list with top 3 best models
        # The best model will be selected, just the name, and a model object will be created based on it
        logger.info(f"Start Lazy predictor")
        if model_name is None:
            model_names = get_model_names(X_train, X_test, y_train, y_test, top_n_models)
            # Taking just column Model from the dataframe
            model_name = model_names['Model'] 

        # This function will compare top n best models and return the best one.
        # that's why is necessary to send trials and splits variables
        logger.info(f"Start selecting model")
        best_model = compare_models(X_train, y_train, X_test_compare, y_test_compare, model_name, hyperparam_search_trials, hyperparam_splits)

        # 9. fit model
        best_model = best_model.fit(X_train, y_train)

        # 10. Validate final model with testing data
        y_predict = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_predict)
        logger.info(f"MAE (mean absolute error) test dataset= {mae}")
        mlflow.log_metric("mae selected model", mae)


        # 11. Push model if it is good enough
        # To know if the trained model is good enough to be load in the model registry it needs to be compared with the baseline model

        # Check of the model is bettern than the best model
        diff_mae = mae - mae_baseline_model
        ratio_diff_mae = diff_mae / mae_baseline_model

        # If ratio of diff represent less than threshold to select model then registry this new model
        if ratio_diff_mae < treshold_select_model:
            # Load model in the registry model mlflow
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="sklearn-model",
                input_example=X_train,
                registered_model_name=f"sk-learn-mode-{experiment_name}",
            )
            logger.info(f"New model registered. Ratio difference with base model: {ratio_diff_mae}")
        else:
            logger.info(f"Model wasn't good enough to be published, it has to be at least 5% underneath MAE of base model")

if __name__ == "__main__":

    settings = Settings()

    train(
        settings.RISINGWAVE_HOST,
        settings.RISINGWAVE_PORT,
        settings.RISINGWAVE_USER,
        settings.RISINGWAVE_DATABASE,
        settings.SYMBOL,
        settings.DAYS_IN_PAST,
        settings.CANDLE_SECONDS,
        settings.PREDICTION_HORIZON_SECONDS,
        settings.N_ROWS_FOR_DATA_PROFILING,
        settings.MLFLOW_TRACKING_URI,
        settings.TRAIN_TEST_SPLIT_RATIO,
        settings.LIST_FEATURES,
        settings.HYPERPARAM_SEARCH_TRIALS,
        settings.HYPERPARAM_SPLITS,
        settings.MODEL_NAME,
        settings.TOP_N_MODELS,
        settings.TRESHOLD_NULL_VALUES,
        settings.TRESHOLD_SELECT_MODEL
    )