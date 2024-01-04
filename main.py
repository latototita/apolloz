from flask import Flask
app = Flask(__name__)
from script import run_trading_bot
symbol='XAGUSDm'
timeframe='1h'


@app.route('/')
def home():
    run_trading_bot()
    return "I am alive"

if __name__=="__main__":
  os.system("python script.py &")
  app.run(host="0.0.0.0", port=80)
