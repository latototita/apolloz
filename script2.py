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
from apollo2 import prediction_home
from get_candles import get_candles_m
data_list=[]
token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjg1NTM4OTg4fQ.K0bb-27iMrcf3gDYGylSgmf1KkcIgnLDL961KBHD3vuYwLC9funTPn-U7wBhvBUDN9pXwdwkBPoA19zIOiZLUxLcNWKcQD3i26TIdu9EhES1xnl1_dLfTPeDhN6SCHGZILh2fO331HexxRa0wqmOiUKYEZgLHSo9VXMCtFSgxJyqrhQzU35U76EWCKHI4yIYRAu8XSFR8RZ6GjeBgqI-J7Y--Z68ldAWisc2RKDUgFeo4ooillmrzTr73dr1usEn9APO25jeUGLm6Qkc8u8eox_vqSvFqovpZZ3czbR21-oEdqFT5EunGh-98WBND6IXfZlxDlBHJ-Ps7r1o9jm4A7vUPBFuGQ6MQ1dcUqKTNYA4p2DGA4lgB1kljoUQhPFau1QkgsJxc7KZExLs8Clg4aNybEO8SwP7uKt9V2UBDqRJT7ZUIrKKgz0uNisuPmS8ml5kKOKcZVQaAUvkbXJuI6vmKWVPeZdGEJu009W-tOuAvgiy2xgrtUpTFBgPAPciK-jrxiRdLHBTij40uYem0UhdmmlaUEH9FGnf9LpnVkvVTl7nrANf3g-yOI3yOAoBupZfAPucEGP8HVvZBfmwdu2GhAMs1cDDij49AUJEoBt1FDqYxOgIyvhGY5Baisn9FC_V-FROyKASzXz0A3cHZUZ63Vm9ghDsDA6rOJd1Kkk'
accountId = os.getenv('ACCOUNT_ID') or '556c93cf-3c24-4095-b576-ed5279fc2d3d'
symbol='EURUSDm'
timeframe='4h'
#symbol_list = ['XAGUSDm', 'EURUSDm', 'GBPUSDm', 'GBPTRYm', 'XAUUSDm', 'AUDCHFm', 'NZDUSDm', 'GBPCHFm', 'USDCHFm','XAUUSDm']
#symbol_list = ['EURUSDm', 'GBPUSDm','GBPCHFm','USDCHFm','AUDCHFm',]

def run_trading_bot():
    async def main():
        while True:
            # Connect to the MetaTrader account
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
            take_profit,min_difference_buy,min_difference_sell,mse=prediction_home(df)
            #print(take_profit,min_difference_buy,min_difference_sell,mse)
            current_price=df['close'].iloc[-1]

            if float(mse)<=0.001:
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
                if len(trades)>1:
                    print(trades)
                    print("There are open trades. Skipping analysis.")
                    await asyncio.sleep(120)
                else:
                    print('passed 1')
                    # Connect to MetaApi API
                    
                    if current_price>take_profit:
                        #take_profit=take_profit+min_difference_sell
                        print('passed 2')
                        stop_loss=current_price +((current_price-take_profit)*3)
                        try:
                            
                            result = await connection.create_market_sell_order(
                                symbol,
                                0.05,
                                stop_loss,
                                take_profit,
                                )
                            print(f'Sell Signal (T)   :Sell Trade successful')
                            
                        except Exception as err:
                            print('Trade failed with error:')
                            print(api.format_error(err))
                    elif current_price<take_profit:
                        #take_profit=take_profit-min_difference_buy
                        print('passed 3')
                        stop_loss=current_price -((take_profit-current_price)*3)
                        try:
                            result = await connection.create_market_buy_order(
                                symbol,
                                0.05,
                                stop_loss,
                                take_profit,
                                )
                            print(f'Buy_Signal (T)   :Buy Trade successful')
                        except Exception as err:
                            print('Trade failed with error:')
                            print(api.format_error(err))

                    
                    
                            
                    current_trades = await connection.get_positions()
                    if len(current_trades)>1:
                        await asyncio.sleep(600)
                    else:
                        print("--------------------------------------------------")


            await asyncio.sleep(300)  # Sleep for 1 minute before the next iteration
    asyncio.run(main())


