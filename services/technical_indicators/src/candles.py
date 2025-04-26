
from config import Settings
from quixstreams import State

def compare_equal_start_time(new_candle: dict, last_historical_candle: dict) -> bool:
    """
    This function compares the new input candle with the last historical candle and checks if they belong to the same time period.
    
    Args:
        new_candle (dict): The new candle to be added or updated
        last_historical_candle (dict): The last candle in the historical data

    Returns:
        bool: True if the new candle has the same start time and symbol as the last historical candle, False otherwise
    """
    return ( 
        (new_candle['window_start_ms'] == last_historical_candle['window_start_ms']) 
        and (new_candle['symbol'] == last_historical_candle['symbol'])
        )

#Add candle to state, save them in memory to calculate technical indicators
def update_candle_in_state(candle: dict, state: State):

    '''
    This is an state function on charge of updating candles in memory.
    Because of sdf.current() the candle service is sending candles without them to finish, it means i'll have repeated candles for a same period of time.
    Then I need to check if a candle belong tom the same period of time, is for a different period o ftime or is simple empty, the first.

    1. I get the State object containning all currenct candles processed
    2. I compare the information of the last candle with the new one going into the process
    3. If the last candle is the same as the new one, I update it
    4. If the last candle is different, I add it to the list

    Args:
        candle (dict): The new candle to be added or updated
        state (State): The state object containing the current candles

    Returns:
        State: The updated state object containing the current candles
    '''
    config = Settings()
    breakpoint()
    candles = state.get('candles', default=[])

    #If there are no candles, add the first one
    if not candles:
        candles.append(candle)

    #Compare and update or add candle
    if compare_equal_start_time(candle, candles[-1]):
        candles[-1] = candle
    else:
        candles.append(candle)

    #Two things are important:
    #1. Kafka save data in memory, it is important to limit the number of candles in memory
    #2. The number of candles in memory will depend on the period of time required by technical indicators

    max_candle_state = config.MAX_CANDLE_STATE
    
    if len(candles) > max_candle_state:
        candles.pop(0)

    #return new state
    state.set('candles', candles)      

    return state
    
