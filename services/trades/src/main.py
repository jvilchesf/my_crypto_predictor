# ### Development Plan
#         - Connect to Kraken socket
#         - Pull transactions
#         - Send transactions to kafka
#             - Create topic
#             - Connect topic
#             - Load topic

from loguru import logger
from quixstreams import Application

from config import config
from kraken_websocket_api import Kraken_WebSocket_API
from kraken_rest_api import Kraken_Rest_API
from model import Trade


def run(
    kafka_host: str,
    kafka_topic_output: str,
    api: Kraken_WebSocket_API | Kraken_Rest_API,
):
    # Create an Application - the main configuration entry point
    app = Application(broker_address=kafka_host, consumer_group="text-splitter-v1")

    # Define a topic with chat messages in JSON format
    messages_topic = app.topic(name=kafka_topic_output, value_serializer="json")

    # create instance trade object where data will be saved
    with app.get_producer() as producer:
        while not api.is_done:
            # query over kraken api on a infinite loop
            events: list[Trade] = api.get_trades()

            for event in events:
                # breakpoint()
                # Serialise trade object to dictionary
                message = messages_topic.serialize(
                    key=event.symbol, value=event.model_dump()
                )
                # 3. Produce a message into the Kafka topic
                producer.produce(
                    topic=messages_topic.name, value=message.value, key=message.key
                )

                # logger.info(f'Produced message to topic {topic.name}')
                logger.info(f"Trade {event.model_dump()} pushed to Kafa")


if __name__ == "__main__":
    logger.info(f"Starting get-trades service in {config.live_or_historical} mode")
    if config.live_or_historical == "live":
        logger.info("Starting live mode")
        # create instance of webosocket kraken api
        api = Kraken_WebSocket_API(config.cryptos_id)
    elif config.live_or_historical == "historical":
        logger.info("Starting historical mode")
        # create instance of rest kraken api
        api = Kraken_Rest_API(config.cryptos_id, config.last_n_days_rest_api)
    else:
        logger.error("Invalid mode")
        raise ValueError("Invalid mode")

    run(config.kafka_host, config.kafka_topic_output, api)
