from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../settings.env")

    RISINGWAVE_HOST: str 
    RISINGWAVE_PORT: int 
    RISINGWAVE_USER: str
    RISINGWAVE_DATABASE: str 

    SYMBOL: str
    DAYS_IN_PAST: int
    CANDLE_SECONDS: int
    PREDICTION_HORIZON_SECONDS: int
    N_ROWS_FOR_DATA_PROFILING: Optional[int] = None

    MLFLOW_TRACKING_URI: str
    TRAIN_TEST_SPLIT_RATIO: float

    LIST_FEATURES: list[str]

    HYPERPARAM_SEARCH_TRIALS: int
    HYPERPARAM_SPLITS: int

    MODEL_NAME: Optional[str] = None
    TOP_N_MODELS: Optional[int] = None

    TRESHOLD_NULL_VALUES: float
    TRESHOLD_SELECT_MODEL: float
    
settings = Settings()