# ### Development Plan
#         - Connect to Kraken socket
#         - Pull transactions
#         - Send transactions to kafka
#             - Create topic
#             - Connect topic
#             - Load topic

from quixstreams import Application
from config import Settings
from api_kraken import Kraken_Api


def run(
        kraken_api: Kraken_Api
    ):
    
    ws = kraken_api.create_connection()

    while 1:
        print(ws.recv())
    

if __name__ == '__main__':
    #Import enviromental variables
    config = Settings()

    #create instance of kraken api
    kraken_api = Kraken_Api(config.cryptos_id)

    # Create an Application - the main configuration entry point
    app = Application(broker_address=config.kafka_host, consumer_group="text-splitter-v1")

    # Define a topic with chat messages in JSON format
    messages_topic = app.topic(name="messages", value_serializer="json")

    run(kraken_api)


