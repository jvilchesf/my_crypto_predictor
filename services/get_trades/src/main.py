# ### Development Plan
#         - Connect to Kraken socket
#         - Pull transactions
#         - Send transactions to kafka
#             - Create topic
#             - Connect topic
#             - Load topic

from quixstreams import Application
from config import Settings
from websocket import create_connection
import json

def run():

    #Kraken socket URL
    kraken_url = "wss://ws.kraken.com/"

    #Kraken subscription info.
    subscription = {
                    "event":"subscribe", 
                    "subscription":{"name":"ticker"},
                    "pair":["BTC/USD"]
                    }
    #create connection with socket                    
    ws = create_connection(kraken_url)

    #Send subscription info. as texr
    subscription = json.dumps(subscription)

    ws.send(subscription)
    
    print(ws.recv())

    print("it has run")
    return 1

if __name__ == '__main__':
    #Import enviromental variables
    config = Settings()
    KAFKA_HOST = config.kafka_host

    # Create an Application - the main configuration entry point
    app = Application(broker_address=KAFKA_HOST, consumer_group="text-splitter-v1")

    # Define a topic with chat messages in JSON format
    messages_topic = app.topic(name="messages", value_serializer="json")
    
    run()

