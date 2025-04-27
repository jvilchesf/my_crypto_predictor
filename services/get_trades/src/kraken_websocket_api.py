from websocket import create_connection
from model import Trade
import json
import loguru

class Kraken_WebSocket_API:

    #Kraken socket URL
    kraken_url = "wss://ws.kraken.com/v2"

    def __init__ (
        self,
        cryptos_id: list[str]
        ):
        
        #create connection with socket          
        try:          
            self.ws_client = create_connection(self.kraken_url)
        except Exception as e:
            loguru.logger.error(f"Error creating connection: {e}")
            raise

        #Create subscription
        try:
            self._subscribe(cryptos_id)           
        except Exception as e:
            loguru.logger.error(f"Error creating subscription: {e}")
            raise

    def get_trades(self) -> list[Trade]:

        try:
            #Get trade from socket
            results = self.ws_client.recv() 
        except:
            loguru.logger.error(f"Error getting trade")
        
        #if result is heartbeat skip
        if "heartbeat" in results:
            return []
        
        # transform raw string into a JSON object
        try:
            data = json.loads(results)
        except json.JSONDecodeError as e:
            logger.error(f'Error decoding JSON: {e}')
            return []

        #Get trades data
        try:
            trades_data = data['data']
        except KeyError as e:
            logger.error(f'No `data` field with trades in the message {e}')
            return []

        #Create empty trade list
        trades_list = []
        #Save trades in a list
        for trade in trades_data:
            new_trade = Trade(
                symbol= trade['symbol'],
                price= trade['price'],
                qty= trade['qty'],
                timestamp= trade['timestamp']                
                )
            trades_list.append(new_trade)    
        
        #Return list of trades
        return trades_list



    def _subscribe (
        self,
        cryptos_id: list[str]
        ):
        #Use library websocket to create a subscription
        self.ws_client.send(json.dumps(
            {
                "method": "subscribe",
                "params": {
                    "channel": "trade",
                    "symbol": cryptos_id,
                    "snapshot": True
                }
            }
        ))

        # discard the first 2 messages for each product_id
        # as they contain no trade data
        for _ in cryptos_id:
            _ = self.ws_client.recv()
            _ = self.ws_client.recv()

        return None
