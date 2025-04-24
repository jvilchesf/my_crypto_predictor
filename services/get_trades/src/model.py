from pydantic import BaseModel
from datetime import datetime

class Trade(BaseModel):
    symbol: str
    price: float
    qty: float
    timestamp: str
