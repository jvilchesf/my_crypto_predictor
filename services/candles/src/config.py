from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file = "services/candles/src/settings.env", env_file_encoding = "utf-8")

    KAFKA_HOST: str
    TIMEFRAME_CANDLE: int

config = Settings()    