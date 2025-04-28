from model import Trade
import requests
import time
import json
import loguru

URL = "https://api.kraken.com/0/public/Trades"

class Kraken_Rest_API:
    def __init__(
        self, 
        cryptos_id:list[str],
        last_n_days_rest_api: int
        ):

        # Initialize variables
        self.last_n_days_rest_api = last_n_days_rest_api

        self.list_cryptos_id = cryptos_id
        self.cryptos_id = self.list_cryptos_id[0]

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
            time.sleep(1) # to avoid too many requests
        except:
            loguru.logger.error(f"Error getting trades")
            time.sleep(10)
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
        # update the since_timestamp_ns
        self.since_timestamp_ns = int(float(data['result']['last']))
        
        #check if is done, if last timestamp pulled from the api is greather than current timestamp then finish
        if self.since_timestamp_ns > int(time.time_ns() - 30 * 1e9):
            #If it is the last trade. I have to be sure now if it is the last symbol or crypto
            # If it is the last then finish and return is done True
            if self.cryptos_id == self.list_cryptos_id[-1]:
                self.is_done = True
            # But if it is not the last it is necessary to re-assign cryptos_id and since_timestamp_ns    
            else:
                self.cryptos_id = self.list_cryptos_id[self.list_cryptos_id.index(self.cryptos_id) + 1]
                self.since_timestamp_ns = int(
                    time.time_ns() - self.last_n_days_rest_api * 24 * 60 * 60 * 1000000000
                )

        return trades_list

    def is_done() -> bool:
        #check if is done
        return False