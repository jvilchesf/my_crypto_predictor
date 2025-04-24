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
from model import Trade
import json


def run(
        kraken_api: Kraken_Api
    ):
    
    #create connection with kraken api
    ws = kraken_api.create_connection()

    #create instance trade object where data will be saved
    trades_list = []
    while True:
        #query over kraken api on a infinite loop
        result = ws.recv()
        #transform str result into a dictionary
        results = json.loads(result)
        #save trade in trades_list
        save_trade(results, trades_list)
 

def save_trade(
    results: dict,
    trades_list: list[Trade]
) -> list:

    #if result[channel] is hearteat skip, if channel = trade save in a object
    if 'channel' in results and results['channel'] == 'trade':
        trades = results['data']
        for trade in trades:
            new_trade = Trade(
                symbol= trade['symbol'],
                side= trade['side'],
                price= trade['price'],
                qty= trade['qty'],
                ord_type= trade['ord_type'],
                trade_id= trade['trade_id'],
                timestamp= trade['timestamp']                
                )
            trades_list.append(new_trade)    
    return trades_list

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


