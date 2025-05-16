from risingwave import RisingWave, RisingWaveConnOptions, OutputFormat
from ydata_profiling import ProfileReport
import great_expectations as gex
import pandas as pd
from loguru import logger
from config import Settings
from names import get_experiment_name

import mlflow
from mlflow.data.pandas_dataset import PandasDataset

from models import BaselineModel, fit_lazy_regresor_n_models

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

def validate_data(
    ts_data: pd.DataFrame
    ):
    """
    Runs a bunch of validations, if any of them fail, the function will raise an exception
    """

    # Check if the numeric columns are positive
    ge = gex.from_pandas(ts_data)

    #validation_results = ge.expect_column_values_to_be_between(column='open', min_value=0, max_value=float('inf'))
    #validation_results = ge.expect_column_values_to_be_between(column='high', min_value=0, max_value=float('inf'))
    #validation_results = ge.expect_column_values_to_be_between(column='low', min_value=0, max_value=float('inf'))
    validation_results = ge.expect_column_values_to_be_between(column='close', min_value=0, max_value=float('inf'))
    #validation_results = ge.expect_column_values_to_be_between(column='volume', min_value=0, max_value=float('inf'))
    
    if not validation_results.success:
        raise Exception("Data validation failed")
    
    # - Check for null values
    # - Check for datetime corrected format
    # - Check for duplicate rows
    # - Check data is sorted by window_start_ms
    
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
    list_features: list[str]
    ):

    """
    train a predictor model forh the give data pair and data, and if the model is good enough it will be saved to the model registry
    """

    logger.info(f"Starting training for {symbol} with {days_in_past} days in the past and {candle_seconds} seconds per candle")

    # 0. Set MLflow 
    # - tracking uri
    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

    # Create a new MLflow Experiment
    mlflow.set_experiment(get_experiment_name(symbol, days_in_past, candle_seconds))

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

        #Drop rows with null values

        ts_data = ts_data.dropna()
        logger.info(f"Dropped rows with null values")

        # Log data to MLflow
        # Wrap dataframe as dataset
        dataset: PandasDataset = mlflow.data.from_pandas(ts_data)
        mlflow.log_input(dataset, context="training")

        # Log dataset size
        mlflow.log_param("ts_data_shape", ts_data.shape)

        # 3. Validate the data  with greate expectations
        logger.info("Validating data with GEX (great expectations) python library")
        validate_data(ts_data)

        # 4. Profile the data using ydata-profiling
        ts_data_for_profiling = ts_data.head(n_rows_for_data_profiling) if n_rows_for_data_profiling else ts_data
        logger.info("Profiling the data using ydata-profiling")
        generate_exploratory_data_analysis(ts_data_for_profiling, output_file_path="./prediction_crypto_eda.html")
        # Log html to MLflow
        mlflow.log_artifact(local_path= "./prediction_crypto_eda.html", artifact_path="eda_report")

        # 5. Split it into train and test
        logger.info(f"Splitting data into train and test with ratio {train_test_split_ratio}")
        
        train_size = int(len(ts_data) * train_test_split_ratio)

        train_data = ts_data.iloc[:train_size]
        test_data = ts_data.iloc[train_size:]
        #Log parameters into mlflow
        mlflow.log_param("train_size", train_data.shape)
        mlflow.log_param("test_size", test_data.shape)

        #Split data into features and target
        X_train = train_data.drop(columns=['target'])
        y_train = train_data['target']
        X_test = test_data.drop(columns=['target'])
        y_test = test_data['target']

        #log parameters into mlflow
        mlflow.log_param("X_train_shape", X_train.shape)
        mlflow.log_param("Y_train_shape", y_train.shape)
        mlflow.log_param("X_test_shape", X_test.shape)
        mlflow.log_param("Y_test_shape", y_test.shape)

        # 6. Baseline model
        logger.info("Building a dummy baseline model")
        baseline_model = BaselineModel()
        y_pred_baseline_model = baseline_model.predict(X_test)

        # 7. Log baseline performance
        # Define an error metric MEA (mean absolute error)
        mae_baseline_model = mean_absolute_error(y_test, y_pred_baseline_model)
        mlflow.log_metric("mae_baseline_model", mae_baseline_model)
        logger.info(f"MAE of the baseline model is {mae_baseline_model}")

        # 7. Train a set of models and see which one perform the best
        # Use lazy predict to evaluate the dataset with a list of models
        df_lazy_predictor = fit_lazy_regresor_n_models(X_train, X_test, y_train, y_test)

        #Reset index to save all columns in mlflow .json
        df_lazy_predictor = df_lazy_predictor.reset_index()

        #Log lazy predictor table in mlflow
        logger.info(f"Saving lazy predictor models performance into mlflow")
        mlflow.log_table(df_lazy_predictor, artifact_file='models_evaluation_lazy_predictor.json')

        print(df_lazy_predictor)

        # 8. XGBoost model with default hyperparameters

        # 9. Log XGBoost performance
        # 10. Validate final model
        # 11. Push model if it is good enough
        

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
        settings.LIST_FEATURES
    )