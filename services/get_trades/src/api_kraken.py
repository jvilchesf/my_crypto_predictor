from websocket import create_connection
import json

class Kraken_Api:

    def __init__ (
        self,
        cryptos_id: list[str]
        ):
    
        #Kraken socket URL
        self.kraken_url = "wss://ws.kraken.com/v2"

        #Kraken subscription info.
        self.subscription_string = {
                        "method": "subscribe",
                        "params": {
                            "channel": "trade",
                            "symbol": cryptos_id,
                            "snapshot": True
                        }
                    }

    def create_connection(self):

        #create connection with socket                    
        ws = create_connection(self.kraken_url)

        #Create subscription
        self._subscribe(ws)

        return ws


    def _subscribe (
        self,
        ws
    ):

        #Use library websocket to create a subscription
        ws.send(json.dumps(
            self.subscription_string
        ))

        return None
