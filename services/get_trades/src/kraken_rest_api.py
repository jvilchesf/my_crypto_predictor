from model import Trade
import requests
import time
import json
import loguru

URL = "https://api.kraken.com/0/public/Trades"

class Kraken_Rest_API:
    def __init__(
        self, 
        cryptos_id:str,
        last_n_days_rest_api: int
        ):

        # Initialize variables
        self.last_n_days_rest_api = last_n_days_rest_api
        self.cryptos_id = cryptos_id

        # Calculate the timestamp for fetching data (in nanoseconds)
        # get current timestamp in nanoseconds
        self.since_timestamp = int(
            time.time_ns() - self.last_n_days_rest_api * 24 * 60 * 60 * 1000000000
        )
            

    def get_trades(self) -> list[Trade]:
        
        trades = []

        # Step 1. Set the right headers and parameters for the request
        headers = {'Accept': 'application/json'}
        params = {
            'pair': self.cryptos_historical_id,
            'since': self.since_timestamp,
        }

        #Get trades from kraken rest api
        while True:
            try:
                response = requests.request("GET", URL, headers=headers, params=params)
            except:
                loguru.logger.error(f"Error getting trades")
                sleep(10)
                raise
            
            #Parse output as a dictionary
            data = json.loads(response.text)
            breakpoint()
            trades = data['result']

        return []