from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
import os
import json

class Settings(BaseSettings):
    # Use environment variable for settings file path, with a default fallback
    model_config = SettingsConfigDict(
        env_file=os.getenv('SETTINGS_FILE', 'settings.env'),
        env_file_encoding='utf-8'
    )

    RISINGWAVE_HOST: str 
    RISINGWAVE_PORT: int 
    RISINGWAVE_USER: str
    RISINGWAVE_DATABASE: str 

    SYMBOL: str
    DAYS_IN_PAST: int
    CANDLE_SECONDS: int
    PREDICTION_HORIZON_SECONDS: int
    N_ROWS_FOR_DATA_PROFILING: Optional[int] = None
    
    # Define the default features as a proper Python list
    LIST_FEATURES: List[str] = [
        "window_start_ms", "open", "high", "low", "close", "target", "volume",
        "sma_7", "sma_14", "sma_21", "sma_60",
        "rsi_7", "rsi_14", "rsi_21", "rsi_60",
        "macd_7", "macdsignal_7", "macdhist_7", "obv_7"
    ]

    MLFLOW_TRACKING_URI: str
    TRAIN_TEST_SPLIT_RATIO: float

    MODEL_NAME: Optional[str] = None
    TOP_N_MODELS: Optional[int] = None

    HYPERPARAM_SEARCH_TRIALS: int
    HYPERPARAM_SPLITS: int
    TRESHOLD_NULL_VALUES: float
    TRESHOLD_SELECT_MODEL: float
    
settings = Settings()