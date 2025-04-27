from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file = "services/technical-indicators/src/settings.env", env_file_encoding = "utf-8")

    KAFKA_HOST: str
    TIMEFRAME_CANDLE: int
    MAX_CANDLE_STATE: int
    LIST_SMA_PERIODS: list[int]
    TOPIC_NAME: str
    RISINGWAVE_TABLE_NAME: str
    CRYPTOS_ID: list[str]