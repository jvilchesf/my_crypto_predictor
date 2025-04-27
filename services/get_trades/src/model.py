from pydantic import BaseModel
from datetime import datetime

class Trade(BaseModel):
    symbol: str
    price: float
    qty: float
    timestamp: str
    timestamp_ms: int

    def to_dict(
        self
        ) -> dict:

        return self.model_dump()

    @classmethod
    def format_trade_object(
        cls,
        symbol: str,
        price: float,
        qty: float,
        timestamp_sec: float
        ) -> 'Trade':

        return cls(
            symbol=symbol,
            price=price,
            qty=qty,
            timestamp=datetime.fromtimestamp(timestamp_sec).strftime('%Y-%m-%d %H:%M:%S'),
            timestamp_ms=int(timestamp_sec * 1000)
        )
            