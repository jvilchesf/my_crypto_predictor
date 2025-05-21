from quixstreams import State
from talib import stream
import numpy as np
from config import Settings

config = Settings()


def calculate_technical_indicators(candle: dict, state: State):
    """
    Compute technical indicators from the candles in the state dictionary
    """

    # Simple moving average
    # Define windows to compute this indicator
    # - windows: 7
    # - windows: 14
    # - windows: 21

    # get list of candles
    candles = state.get("candles", [])
    # get list of close prices and transform it into a numpy array, bacause this is the type that talib spect
    # open = np.array([c["open"] for c in candles])
    # high = np.array([c["high"] for c in candles])
    # low = np.array([c["low"] for c in candles])
    close = np.array([c["close"] for c in candles])
    volume = np.array([c["volume"] for c in candles])

    indicators = {}

    # Read list of periods from enviromental variables
    list_sma_periods = config.LIST_SMA_PERIODS
    # Compute moving average different periods
    indicators["sma_7"] = stream.SMA(close, timeperiod=list_sma_periods[0])
    indicators["sma_14"] = stream.SMA(close, timeperiod=list_sma_periods[1])
    indicators["sma_21"] = stream.SMA(close, timeperiod=list_sma_periods[2])
    indicators["sma_60"] = stream.SMA(close, timeperiod=list_sma_periods[3])

    # Exponential moving average
    indicators["ema_7"] = stream.EMA(close, timeperiod=list_sma_periods[0])
    indicators["ema_14"] = stream.EMA(close, timeperiod=list_sma_periods[1])
    indicators["ema_21"] = stream.EMA(close, timeperiod=list_sma_periods[2])
    indicators["ema_60"] = stream.EMA(close, timeperiod=list_sma_periods[3])

    # RSI
    indicators["rsi_7"] = stream.RSI(close, timeperiod=list_sma_periods[0])
    indicators["rsi_14"] = stream.RSI(close, timeperiod=list_sma_periods[1])
    indicators["rsi_21"] = stream.RSI(close, timeperiod=list_sma_periods[2])
    indicators["rsi_60"] = stream.RSI(close, timeperiod=list_sma_periods[3])

    # MACD
    indicators["macd_7"], indicators["macdsignal_7"], indicators["macdhist_7"] = (
        stream.MACD(
            close,
            fastperiod=list_sma_periods[0],
            slowperiod=list_sma_periods[1],
            signalperiod=list_sma_periods[2],
        )
    )

    # #OBV On balance volume indicator
    indicators["obv_7"] = stream.OBV(close, volume)

    # breakpoint()

    return {**candle, **indicators}
