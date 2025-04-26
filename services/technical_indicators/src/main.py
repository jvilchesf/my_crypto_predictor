from quixstreams import Application
from config import Settings
from candles import update_candle_in_state
from technical_indicators import calculate_technical_indicators

#This process has the role of receiving candles from the kafka topic "candles", and calculate new technical indicators based ont it

# Technical indicators to calculate:
#   [Moving average]
    #   Moving average = Window aggregation based on time to calculate the average of the price. It could change depending on how windows is defined
    #   Exponensial
#    [Momentum indicator]
    #   RSI
    #   MACD = Try to capture momentum. What it does is calculate something (average, volatiliy, etc.) in a group of candles.
#   [Volume indicator] 
    #   OBV 

def run(
    kafka_host: str,
    timeframe_candle: int
    ):

    #Create an application
    app = Application(broker_address= kafka_host, consumer_group="technical-indicators-v1", auto_offset_reset='earliest')

    #Create topic where the information is
    topic_input = app.topic(name = 'trades', value_deserializer="json")

    #Create topic where the information is
    topic_output = app.topic(name = 'technical-indicators', value_deserializer='json')

    #Create dataframe
    sdf = app.dataframe(topic_input) 

    #Filter candles based on time, there are different windows time candles. 
    #Filter based on "candle_seconds"
    sdf = sdf[
        sdf['candle_seconds'] == 60
    ]

    #Add candles to the state dictionary
    sdf = sdf.apply(update_candle_in_state, stateful=True)
    
    #Calculate technical indicators
    sdf = sdf.apply(calculate_technical_indicators, stateful=True)

if __name__ == '__main__':

    #Import enviroment variables
    config = Settings()

    run(config.KAFKA_HOST,
        config.TIMEFRAME_CANDLE
        )