from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file = "services/get_trades/src/settings.env", env_file_encoding = "utf-8")

    kafka_host: str
    cryptos_id: List[str]
    live_or_historical: Literal['live','historical'] = 'live'
    last_n_days_rest_api: int
    
config = Settings()