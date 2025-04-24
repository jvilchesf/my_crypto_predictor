from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file = "services/get_trades/src/settings.env", env_file_encoding = "utf-8")

    kafka_host: str
    cryptos_id: list = ['BTC/USD',
                        'ETH/USD'
                    ]

config = Settings()