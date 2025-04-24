from pydantic import BaseModel
from datetime import datetime

class Trade(BaseModel):
    symbol: str
    side: str
    price: float
    qty: float
    ord_type: str
    trade_id: int
    timestamp: str
