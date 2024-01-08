from flask import Flask, request
from script import run_trading_bot
import threading
import time

app = Flask(__name__)

# Variable to track whether the thread is still running
is_trading_bot_running = False

# Function to run the trading bot in a separate thread
def run_trading_bot_thread():
    global is_trading_bot_running
    is_trading_bot_running = True
    start_time = time.time()
    
    # Run the trading bot
    run_trading_bot()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Trading bot execution time: {elapsed_time} seconds")
    
    # Update the flag to indicate that the thread has completed
    is_trading_bot_running = False


# Route to initiate the trading bot
@app.route('/')
def initiate_trading_bot():
    global is_trading_bot_running
    
    # Check if the trading bot is already running
    if is_trading_bot_running:
        return "Trading bot is already running."
    
    # Start the trading bot in a separate thread
    thread = threading.Thread(target=run_trading_bot_thread)
    thread.start()
    
    return "Trading bot initiation requested successfully!"

if __name__ == '__main__':
    app.run(debug=True)
