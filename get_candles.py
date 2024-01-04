import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
#from sklearn.ensemble import forest
from metaapi_cloud_sdk import MetaApi
from metaapi_cloud_sdk.clients.metaApi.tradeException import TradeException
from datetime import datetime, timedelta
from metaapi_cloud_sdk import MetaApi
from datetime import datetime, timedelta
import os
import asyncio
import pandas as pd

token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwiYWNjZXNzUnVsZXMiOlt7ImlkIjoidHJhZGluZy1hY2NvdW50LW1hbmFnZW1lbnQtYXBpIiwibWV0aG9kcyI6WyJ0cmFkaW5nLWFjY291bnQtbWFuYWdlbWVudC1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoibWV0YWFwaS1yZXN0LWFwaSIsIm1ldGhvZHMiOlsibWV0YWFwaS1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoibWV0YWFwaS1ycGMtYXBpIiwibWV0aG9kcyI6WyJtZXRhYXBpLWFwaTp3czpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoibWV0YWFwaS1yZWFsLXRpbWUtc3RyZWFtaW5nLWFwaSIsIm1ldGhvZHMiOlsibWV0YWFwaS1hcGk6d3M6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFzdGF0cy1hcGkiLCJtZXRob2RzIjpbIm1ldGFzdGF0cy1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoicmlzay1tYW5hZ2VtZW50LWFwaSIsIm1ldGhvZHMiOlsicmlzay1tYW5hZ2VtZW50LWFwaTpyZXN0OnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJjb3B5ZmFjdG9yeS1hcGkiLCJtZXRob2RzIjpbImNvcHlmYWN0b3J5LWFwaTpyZXN0OnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJtdC1tYW5hZ2VyLWFwaSIsIm1ldGhvZHMiOlsibXQtbWFuYWdlci1hcGk6cmVzdDpkZWFsaW5nOio6KiIsIm10LW1hbmFnZXItYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX1dLCJ0b2tlbklkIjoiMjAyMTAyMTMiLCJpbXBlcnNvbmF0ZWQiOmZhbHNlLCJyZWFsVXNlcklkIjoiNmIyNDU0NGVmMzFiNDc0ODVjMTc0NTZlMzc3ZmE5YWYiLCJpYXQiOjE2OTgzMTkwODl9.Aokl2AYcsmtooCAya1LcRrJQNqYaLOO7ezic5HDmdLz9Kp05JkZkk2NoqTXGtePHIf8gxtr9iL6YnG5jBM2VbPVXWEXBKVauMy7Pb5NbpK3avNGQXADj2JCqRhYBYmN0-0l55PTZo8nm87RX_jTR-R2cL3A4AaVtWmoj-jR2c7TDx9mrOWp-fEK32j3Yg2EUT0mW_Ve868TL6x3oQduY4gi8bWMYYYnAicbMgxZtzVa5xyB92eKRiswjcjK4pdz2W_7Zi0HrST8B32GvGcI0SvAz3VVtfcqayT_QGeGHJoX2g722FacWpxbPe5IyShlXXcJE3D82VWISQXcoFn6D3X_Bd7n9ObKalRwF2QvKnVwTuSMM8YmWZJiNheBwieHj0oVW5SoY00BfbGV19ceH4l_j10d24T83o3ZMivwJINTkfO0-pWu6dXPUEWLE0yeJheMk1YJQeJ6Mft5NV902coSnXeeL8fPP_1jM7dUi7WQqVIn8YpgLWCsYujNBW_IecE23iDePLx7kcYE0DP7vvu2Z5LQq9xjmP10Qvqiek8htWOrc2kS0sHJX3bJgTx09zul8YVRjq9N_RegNKgC7-Kvk03aYLMaZO3iBcNLEW7Aj5eWC6cyaS4vcGjsNtKtWdeBiVKcNX-G6MmFhm9Qo_XqReLsZstnR7QW9RHqEwlg'
accountId = os.getenv('ACCOUNT_ID') or '97f2db9a-42f1-407c-90d1-b1382d5d4091'

#symbol='XAGUSDm'['XAGUSDm','XAUUSDm', 'EURUSDm', 'GBPUSDm', 'GBPTRYm','AUDCHFm', 'NZDUSDm',
symbol_list = ['USDJPYm','USDCHFm','EURJPYm','XAUUSDm','XAGUSDm']
#symbol_list = ['XAUUSDm']#,'EURJPYm','XAUUSDm','XAGUSDm']

   
async def get_candles_m(timeframe,symbol):
    api = MetaApi(token)
    account = await api.metatrader_account_api.get_account(accountId)
    initial_state = account.state
    deployed_states = ['DEPLOYING', 'DEPLOYED']
    timeframe=timeframe
    symbol=symbol
    if initial_state not in deployed_states:
        # wait until account is deployed and connected to broker
        print('Deploying account')
        await account.deploy()
        print('Waiting for API server to connect to broker (may take a few minutes)')
        await account.wait_connected()
        
    try:
        # Create an empty dataframe to store the candlestick data
        df = pd.DataFrame()


        # retrieve last 10K 1m candles
        pages = 4
        print(f'Downloading {pages}K latest candles for {symbol}')
        started_at = datetime.now().timestamp()
        start_time = None
        candles = None
        for i in range(pages):
            # the API to retrieve historical market data is currently available for G1 only
            candles = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=start_time)
            print(f'Downloaded {len(candles) if candles else 0} historical candles for {symbol}')
            
            
            if not candles:
                pass
            else:
                #start_time = candles[0]['time']
                #start_time.replace(minute=start_time.minute - 1)
                #print(f'First candle time is {start_time}')
                
                #Create a new dataframe for each iteration and add it to the main dataframe
                new_df = pd.DataFrame(candles)
                df = pd.concat([df, new_df], ignore_index=True)
                print(f'Candles added to dataframe')
                #print(df)
            # Sort the DataFrame based on the 'time' column in descending order
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values(by='time', ascending=True)

        return df

    except Exception as e:
        print("Error retrieving candle data: {e}")
        return f"Error retrieving candle data: {e}"