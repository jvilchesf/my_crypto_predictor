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
from loguru import logger

def run(
        kafka_host: str,
        kraken_api: Kraken_Api
    ):

    # Create an Application - the main configuration entry point
    app = Application(broker_address= kafka_host, consumer_group="text-splitter-v1")

    # Define a topic with chat messages in JSON format
    messages_topic = app.topic(name="trades", value_serializer="json")

    #create instance trade object where data will be saved
    with app.get_producer() as producer:
        while True:
            #query over kraken api on a infinite loop
            events : list[Trade] = kraken_api.get_trades() 
            
            for event in events:
                #breakpoint()
                # Serialise trade object to dictionary
                message = messages_topic.serialize(key=event.symbol, value=event.model_dump())
                # 3. Produce a message into the Kafka topic
                producer.produce(topic=messages_topic.name, value=message.value, key=message.key)

                # logger.info(f'Produced message to topic {topic.name}')
                logger.info(f'Trade {event.model_dump()} pushed to Kafa')
        

if __name__ == '__main__':
    #Import enviromental variables
    config = Settings()

    KAFKA_HOST = config.kafka_host
    CRYPTOS_ID = config.cryptos_id

    #create instance of kraken api
    kraken_api = Kraken_Api(CRYPTOS_ID)

    run(KAFKA_HOST,kraken_api)


