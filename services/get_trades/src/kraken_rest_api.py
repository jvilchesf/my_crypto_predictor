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

        # Initialize flags to control process end
        self.is_done = False

        # Calculate the timestamp for fetching data (in nanoseconds)
        # get current timestamp in nanoseconds
        self.since_timestamp_ns = int(
            time.time_ns() - self.last_n_days_rest_api * 24 * 60 * 60 * 1000000000
        )
            

    def get_trades(self) -> list[Trade]:
        
        trades = []

        # Step 1. Set the right headers and parameters for the request
        headers = {'Accept': 'application/json'}
        params = {
            'pair': self.cryptos_id,
            'since': self.since_timestamp_ns,
        }

        #Step 2. Get trades from kraken rest api
        try:
            response = requests.request("GET", URL, headers=headers, params=params)
            sleep(1) # to avoid too many requests
        except:
            loguru.logger.error(f"Error getting trades")
            sleep(10)
            raise
        
        #Step 3. Parse output as a dictionary
        data = json.loads(response.text)
               # Check if 'result' key exists in the response
        if 'result' not in data:
            loguru.logger.error(f"Error in API response: {data}")
            # Sleep to avoid hitting rate limits
            time.sleep(10)
            return []
        else:
            trades = data['result'][self.cryptos_id]

        #Step 4. Transform trades in the list on a Trade object and create a list of them
        trades_list = [ 
                Trade.format_trade_object(
                    symbol = self.cryptos_id,
                    price = trade[0],
                    qty = trade[1],
                    timestamp_sec = trade[2]
                )
            for trade in trades]
        
        #Step 5. Validate when is done
        #Get last trade timestamp
        self.since_timestamp_ns = data['result']['last']

        #get time now in nanoseconds
        time_now_ns = time.time_ns()
        
        #check if is done, if last timestamp pulled from the api is greather than current timestamp then finish
        if int(self.since_timestamp_ns)> time_now_ns:
            self.is_done = True

        return trades_list

    def is_done() -> bool:
        #check if is done
        return False