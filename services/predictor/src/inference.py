from loguru import logger
from config import SettingsInference
from names import get_experiment_name
from model_registry import load_mode_from_registry
import mlflow.sklearn
from risingwave import RisingWave, RisingWaveConnOptions, OutputFormat
import pandas as pd
import threading
import time
from datetime import datetime, timezone

def inference(
            risingwave_host: str,
            risingwave_port: int, 
            risingwave_user: str, 
            risingwave_password: str,
            risingwave_database: str, 
            risingwave_schema: str,
            symbol: str, 
            candle_seconds: int, 
            prediction_horizon_seconds: int, 
            mlflow_tracking_uri: str,
            model_version: str,
            days_in_past: int,
            risingwave_input_table: str,
            risingwave_output_table: str):  # Added output table parameter
    
    # Load last model from model registry
    model_name = get_experiment_name(symbol, days_in_past, candle_seconds)
    model_version = model_version
    model, features = load_mode_from_registry(model_name, model_version, mlflow_tracking_uri)

    # Connect to RisingWave instance
    rw = RisingWave(
        RisingWaveConnOptions.from_connection_info(
            host=risingwave_host, 
            port=risingwave_port, 
            user=risingwave_user, 
            password=risingwave_password, 
            database=risingwave_database
        )
    )

    def prediction_handler(data: pd.DataFrame):
        """
        Handle new data from RisingWave and make predictions
        """
        logger.info(f'Received {data.shape[0]} updates from {risingwave_input_table}')

        # Filter only Insert and Updates
        data = data[data['op'].isin(['Insert', 'UpdateInsert'])]
        # for the given `pair`
        data = data[data['symbol'] == symbol]
        # for the given `candle_seconds`
        data = data[data['candle_seconds'] == candle_seconds]

        # filter old rows
        current_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        data = data[data['window_start_ms'] > current_ms - 1000 * candle_seconds * 2]
        
        if data.shape[0] == 0:
            logger.info(f'No rows to predict for {symbol} {candle_seconds}')
            return
        
        # predict 
        logger.info(f"Predicting")
        logger.info(f"\n{data}")
        predictions = model.predict(data[features])

        # add predictions to data
        data['predictions'] = predictions

        # Create output dataframe
        output = pd.DataFrame()  
        output['price'] = data['close']
        output['predictions'] = predictions  
        output['symbol'] = symbol

        output['prediction_timestamp_ms'] = int(datetime.now(timezone.utc).timestamp() * 1000)
        output['model_name'] = model_name
        output['model_version'] = model_version

        output['predicted_ts_ms'] = (
            data['window_start_ms']
            + (candle_seconds + prediction_horizon_seconds) * 1000
        ).to_list()

        logger.info(f"Output")
        logger.info(f"\n{output}")

        logger.info(f"Writing to {risingwave_output_table}")
        # write to output table
        # Write dataframe to the `risingwave_output_table`
        rw.insert(table_name=risingwave_output_table, data=output)

        
    rw.on_change(
        subscribe_from=risingwave_input_table,
        schema_name=risingwave_schema,
        handler=prediction_handler,
        output_format=OutputFormat.DATAFRAME,
    )

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping inference...")

if __name__ == "__main__":
    settings = SettingsInference()

    inference(
        settings.RISINGWAVE_HOST, 
        settings.RISINGWAVE_PORT,
        settings.RISINGWAVE_USER,
        settings.RISINGWAVE_PASSWORD,
        settings.RISINGWAVE_DATABASE,
        settings.RISINGWAVE_SCHEMA,
        settings.SYMBOL,
        settings.CANDLE_SECONDS,
        settings.PREDICTION_HORIZON_SECONDS,
        settings.MLFLOW_TRACKING_URI,
        settings.MODEL_VERSION, 
        settings.DAYS_IN_PAST,
        settings.RISINGWAVE_INPUT_TABLE,
        settings.RISINGWAVE_OUTPUT_TABLE  # Added output table
    )