from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="services/technical-indicators/src/settings.env",
        env_file_encoding="utf-8",
    )

    KAFKA_HOST: str
    LIST_SMA_PERIODS: list[int]
    RISINGWAVE_TABLE_NAME: str
    KAFKA_TOPIC_INPUT: str
    KAFKA_TOPIC_OUTPUT: str
    MAX_CANDLE_STATE: int
    TIMEFRAME_CANDLE: int
