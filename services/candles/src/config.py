from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="services/candles/src/settings.env", env_file_encoding="utf-8"
    )

    KAFKA_HOST: str
    KAFKA_TOPIC_INPUT: str
    KAFKA_TOPIC_OUTPUT: str
    TIMEFRAME_CANDLE: int
    CRYPTOS_ID: list[str]

config = Settings()
