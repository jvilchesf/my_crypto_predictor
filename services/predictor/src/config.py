from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class SettingsTraining(BaseSettings):
    RISINGWAVE_HOST: str = "localhost"
    RISINGWAVE_PORT: int = 4567
    RISINGWAVE_USER: str = "root"
    RISINGWAVE_DATABASE: str = "dev"

    SYMBOL: str = "BTC/USD"
    DAYS_IN_PAST: int = 30
    CANDLE_SECONDS: int = 60
    PREDICTION_HORIZON_SECONDS: int = 300
    N_ROWS_FOR_DATA_PROFILING: Optional[int] = 100

    # Define the default features as a proper Python list
    LIST_FEATURES: List[str] = [
        "window_start_ms",
        "open",
        "high",
        "low",
        "close",
        "target",
        "volume",
        "sma_7",
        "sma_14",
        "sma_21",
        "sma_60",
        "rsi_7",
        "rsi_14",
        "rsi_21",
        "rsi_60",
        "macd_7",
        "macdsignal_7",
        "macdhist_7",
        "obv_7",
    ]

    MLFLOW_TRACKING_URI: str = "http://localhost:8889"
    TRAIN_TEST_SPLIT_RATIO: float = 0.8

    MODEL_NAME: Optional[str] = None
    TOP_N_MODELS: Optional[int] = 6

    HYPERPARAM_SEARCH_TRIALS: int = 100
    HYPERPARAM_SPLITS: int = 5
    TRESHOLD_NULL_VALUES: float = 0.05
    TRESHOLD_SELECT_MODEL: float = 0.1

class SettingsInference(BaseSettings):

    RISINGWAVE_HOST: str = "localhost"
    RISINGWAVE_PORT: int = 4567
    RISINGWAVE_USER: str = "root"
    RISINGWAVE_PASSWORD: str = ""
    RISINGWAVE_DATABASE: str = "dev"
    RISINGWAVE_INPUT_TABLE: str = "technical_indicators"
    RISINGWAVE_SCHEMA: str = "public"
    SYMBOL: str = 'BTC/USD'
    CANDLE_SECONDS: int = 60
    PREDICTION_HORIZON_SECONDS: int = 300
    DAYS_IN_PAST: int = 30

    MLFLOW_TRACKING_URI: str = "http://localhost:8889"
    MODEL_VERSION: str = "latest"
    RISINGWAVE_OUTPUT_TABLE: str = "predictors"

settings = SettingsTraining()
