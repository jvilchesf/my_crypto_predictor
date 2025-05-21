def get_experiment_name(symbol: str, days_in_past: int, candle_seconds: int) -> str:
    """
    Generate a unique experiment name based on the symbol, days in the past, and candle seconds
    """
    return f"{symbol}_days_{days_in_past}_candle_seconds_{candle_seconds}"
