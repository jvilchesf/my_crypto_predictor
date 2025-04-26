from quixstreams import Application
from config import Settings
from datetime import timedelta

def init_candle(trade:dict):
    '''
    Function to initialize candle with first trade values
    '''

    candle = {
        'open': trade['price'],
        'close': trade['price'],
        'high': trade['price'],
        'low': trade['price'],
        'volume': trade['qty'],
        'symbol': trade['symbol']
    }

    return candle

def update_candle(candle:dict,
    trade:dict) -> dict:
    '''
    Function to update candle with new trade values
    '''

    updated_candle = {
        'open': candle['open'],
        'close': trade['price'],
        'high': max(candle['high'], trade['price']),
        'low': min(candle['low'], trade['price']),
        'volume': candle['volume'] + trade['qty'],
        'symbol': candle['symbol']
    }

    return updated_candle

def run(
    kafka_host: str,
    timeframe_candle: int
):
    
    # Create an Application - the main configuration to connect to a kafka cluster
    app = Application(broker_address=kafka_host, consumer_group="candles-v1", auto_offset_reset='earliest')

    # Define a topic input and deserializer
    topic_input = app.topic(name='trades', value_deserializer='json')

    # Define a topic output and deserializer
    topic_output = app.topic(name='candles', value_deserializer='json')

    # Create a stream channel to receive data in a pandas-like dataframe
    sdf = app.dataframe(topic_input)

    sdf = (
    # Define a tumbling window of 10 seconds
    sdf.tumbling_window(timedelta(seconds=timeframe_candle))
    # Calculate the minimum temperature 
    .reduce(reducer=update_candle, initializer=init_candle)
    )

    # we emit all intermediate candles to make the system more responsive
    sdf = sdf.current()

    # Extract open, high, low, close, volume, timestamp_ms, pair from the dataframe
    sdf['open'] = sdf['value']['open']
    sdf['high'] = sdf['value']['high']
    sdf['low'] = sdf['value']['low']
    sdf['close'] = sdf['value']['close']
    sdf['volume'] = sdf['value']['volume']
    # sdf['timestamp_ms'] = sdf['value']['timestamp_ms']
    sdf['symbol'] = sdf['value']['symbol']

    # Extract window start and end timestamps
    sdf['window_start_ms'] = sdf['start']
    sdf['window_end_ms'] = sdf['end']

        # keep only the relevant columns
    sdf = sdf[
        [
            'symbol',
            # 'timestamp_ms',
            'open',
            'high',
            'low',
            'close',
            'volume',
            'window_start_ms',
            'window_end_ms',
        ]
    ]

    sdf['candle_seconds'] = timeframe_candle

    # Print the input data
    sdf = sdf.update(lambda message: print(f"Input:  {message}"))

    # Send the output data
    sdf = sdf.to_topic(topic_output)

    # Start the stream processing
    app.run(sdf)

if __name__ == '__main__':

    #Import enviromental variables
    config = Settings()

    run(config.KAFKA_HOST,
        config.TIMEFRAME_CANDLE
        )