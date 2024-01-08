import pandas as pd
import numpy as np
import asyncio
import numpy as np
import pandas as pd
import numpy as np
import os
from decimal import Decimal
import pandas_ta as ta
from metaapi_cloud_sdk import MetaApi
import csv
import time
from apollo import prediction_home
from get_candles import get_candles_m
data_list=[]
token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjg1NTM4OTg4fQ.K0bb-27iMrcf3gDYGylSgmf1KkcIgnLDL961KBHD3vuYwLC9funTPn-U7wBhvBUDN9pXwdwkBPoA19zIOiZLUxLcNWKcQD3i26TIdu9EhES1xnl1_dLfTPeDhN6SCHGZILh2fO331HexxRa0wqmOiUKYEZgLHSo9VXMCtFSgxJyqrhQzU35U76EWCKHI4yIYRAu8XSFR8RZ6GjeBgqI-J7Y--Z68ldAWisc2RKDUgFeo4ooillmrzTr73dr1usEn9APO25jeUGLm6Qkc8u8eox_vqSvFqovpZZ3czbR21-oEdqFT5EunGh-98WBND6IXfZlxDlBHJ-Ps7r1o9jm4A7vUPBFuGQ6MQ1dcUqKTNYA4p2DGA4lgB1kljoUQhPFau1QkgsJxc7KZExLs8Clg4aNybEO8SwP7uKt9V2UBDqRJT7ZUIrKKgz0uNisuPmS8ml5kKOKcZVQaAUvkbXJuI6vmKWVPeZdGEJu009W-tOuAvgiy2xgrtUpTFBgPAPciK-jrxiRdLHBTij40uYem0UhdmmlaUEH9FGnf9LpnVkvVTl7nrANf3g-yOI3yOAoBupZfAPucEGP8HVvZBfmwdu2GhAMs1cDDij49AUJEoBt1FDqYxOgIyvhGY5Baisn9FC_V-FROyKASzXz0A3cHZUZ63Vm9ghDsDA6rOJd1Kkk'
accountId = os.getenv('ACCOUNT_ID') or 'bb7fcbbb-5eff-4805-92ca-2369aeb26ee3'
symbol='XAGUSDm'
timeframe='1h'
#symbol_list = ['XAGUSDm', 'EURUSDm', 'GBPUSDm', 'GBPTRYm', 'XAUUSDm', 'AUDCHFm', 'NZDUSDm', 'GBPCHFm', 'USDCHFm','XAUUSDm']
#symbol_list = ['EURUSDm', 'GBPUSDm','GBPCHFm','USDCHFm','AUDCHFm',]

def run_trading_bot():
    async def main():
        # Connect to the MetaTrader account
        
        api = MetaApi(token)
        account = await api.metatrader_account_api.get_account(accountId)
        initial_state = account.state
        deployed_states = ['DEPLOYING', 'DEPLOYED']
        if initial_state not in deployed_states:
            # Wait until the account is deployed and connected to the broker
            print('Deploying account')
            await account.deploy()
        print('Waiting for API server to connect to the broker (may take a few minutes)')
        await account.wait_connected()

        # Connect to MetaApi API
        connection = account.get_rpc_connection()
        await connection.connect()

        # Wait until terminal state synchronized to the local state
        print('Waiting for SDK to synchronize to terminal state (may take some time depending on your history size)')
        await connection.wait_synchronized()
        

        trades = await connection.get_positions()#connection.get_orders()
        if len(trades)>2:
            print(trades)
            print("There are open trades. Skipping analysis.")
            await asyncio.sleep(120)
        else:
            trades = await connection.get_positions()

            if len(trades)>2:
                print("There are open trades. Skipping analysis.")
                await asyncio.sleep(120)
            try:
                prices = await connection.get_symbol_price(symbol)
            except:
                prices=None
            if prices!=None:
                
                if 1<100:
                    try:
                        # Fetch historical price data
                        candles = await get_candles_m(timeframe,symbol)
                        k=True
                        word1="Error"
                        while k==True:
                            if type(candles)== str:
                                if word1.lower() in candles.lower():
                                    candles =await get_candles_m(timeframe,symbol)
                            else:
                                k=False
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")  
                    df = pd.DataFrame(candles)
                    df=df.dropna()
                    # Convert 'time' to datetime and set it as the index
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                    direction,take_profit,accuracy_classifier=prediction_home(df)
                    current_price=df['close'].iloc[-1]
                    #print(df.iloc[-1])
                    print(accuracy_classifier)
                    print(type(accuracy_classifier))
                    accuracy_classifier=float(accuracy_classifier)
                    print(accuracy_classifier)
                    print(type(accuracy_classifier))
                    if accuracy_classifier>=0.93:
                        print('passed 1')
                        if direction=='Sell' and current_price>take_profit:
                            print('passed 2')
                            stop_loss=current_price +((current_price-take_profit)*2)
                            try:
                                
                                result = await connection.create_market_sell_order(
                                    symbol,
                                    0.03,
                                    stop_loss,
                                    take_profit,
                                    )
                                print(f'Sell Signal (T)   :Sell Trade successful, {symbol} : {timeframez}:Margin =,result code is ' + result['stringCode'])
                                
                            except Exception as err:
                                print('Trade failed with error:')
                                print(api.format_error(err))
                        elif direction=='Buy'and current_price<take_profit:
                            print('passed 3')
                            stop_loss=current_price -((take_profit-current_price)*2)
                            try:
                                result = await connection.create_market_buy_order(
                                    symbol,
                                    0.03,
                                    stop_loss,
                                    take_profit,
                                    )
                                print(f'Buy_Signal (T)   :Buy Trade successful, {symbol} : {timeframez} ,result code is ' + result['stringCode'])
                            except Exception as err:
                                print('Trade failed with error:')
                                print(api.format_error(err))

            
            
                    
            current_trades = await connection.get_positions()
            if len(current_trades)>2:
                await asyncio.sleep(120)
            else:
                print("--------------------------------------------------")


        await asyncio.sleep(120)  # Sleep for 1 minute before the next iteration
    asyncio.run(main())


#run_trading_bot()
